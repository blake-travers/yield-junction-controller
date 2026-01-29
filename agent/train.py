import os
import sys
import torch
import numpy as np
import traci
import sumolib
from collections import deque

from agent.agent import DoubleDQNAgent, PrioritizedReplayBuffer
from agent.vehicle import Vehicle
from agent.adjacency_matrix import get_adjacency_matrices
from agent.generate_phases import generate_rule_based_phases, create_phase_mask

SUMO_CFG = "environment/sim.sumocfg"
NET_FILE = "environment/basic_intersection.net.xml"
SUMO_CMD = ["sumo-gui", "-c", SUMO_CFG] # "sumo-gui" to watch / "sumo" to train

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 500
MEMORY_SIZE = 50000
GREEN_DURATION = 7
YELLOW_DURATION = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SumoIntersectionEnv:
    def __init__(self, net_file, sumo_cmd, phases, lane_list):
        self.net_file = net_file
        self.sumo_cmd = sumo_cmd
        self.phases = phases
        self.lane_list = lane_list
        self.active_vehicles = {}
        self.tls_id = None
        
    def reset(self):
        try: traci.close()
        except: pass
        
        traci.start(self.sumo_cmd)
        self.tls_id = traci.trafficlight.getIDList()[0]
        self.active_vehicles = {}
        
        for _ in range(10):
            self._sim_step()
            
        return self.get_state()

    def step(self, action_idx):
        target_phase = self.phases[action_idx]
        current_phase = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        
        if target_phase != current_phase:
            y_state = list(current_phase)
            for i in range(len(current_phase)):
                if current_phase[i].lower() == 'g' and target_phase[i].lower() == 'r':
                    y_state[i] = 'y'
            y_str = "".join(y_state)
            
            traci.trafficlight.setRedYellowGreenState(self.tls_id, y_str)
            for _ in range(YELLOW_DURATION):
                self._sim_step()
        
        traci.trafficlight.setRedYellowGreenState(self.tls_id, target_phase)
        
        for _ in range(GREEN_DURATION):
            self._sim_step()
            
        next_state = self.get_state()
        reward = self.get_reward()
        done = False
        
        return next_state, reward, done

    def _sim_step(self):
        traci.simulationStep()
        
        current_ids = set(traci.vehicle.getIDList())
        
        for vid in current_ids:
            if vid not in self.active_vehicles:
                self.active_vehicles[vid] = Vehicle(vid)
            self.active_vehicles[vid].update()
            
        departed = set(self.active_vehicles.keys()) - current_ids
        for vid in departed:
            del self.active_vehicles[vid]

    def get_state(self):
        features = []
        
        for lane_id in self.lane_list:
            lane_cars = [v for v in self.active_vehicles.values() if v.getLaneID() == lane_id]
            
            queue_count = len(lane_cars)
            
            if queue_count > 0:
                avg_speed = np.mean([v.getSpeed() for v in lane_cars])
                max_wait = np.max([v.wait_time for v in lane_cars])
            else:
                avg_speed = 13.0
                max_wait = 0
            
            features.append([queue_count, avg_speed, max_wait, 0])

        return torch.tensor([features], dtype=torch.float32).to(DEVICE)

    def get_reward(self):

        total_queue = len(self.active_vehicles)
        total_wait = sum([v.wait_time for v in self.active_vehicles.values()])
        
        reward = - (total_queue + (0.1 * total_wait))
        return reward
    

if __name__ == "__main__":
    print("Loading Graph Aadjacency matrices...")
    nodes, adj_flow, adj_conf = get_adjacency_matrices(NET_FILE)
    
    print("Generating valid Traffic light phases...")
    phases = generate_rule_based_phases(NET_FILE)
    phase_mask = create_phase_mask(NET_FILE, phases, nodes)
    
    print("Setting up Intersection Environment...")
    env = SumoIntersectionEnv(NET_FILE, SUMO_CMD, phases, nodes)
    
    agent = DoubleDQNAgent(num_lanes=len(nodes), num_phases=len(phases), input_dim=4, adj_flow=adj_flow, adj_conf=adj_conf, phase_mask=phase_mask, device=DEVICE)
    memory = PrioritizedReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPS_START
    step_count = 0
    
    print(f"Starting Training on {DEVICE}...")
    
    for episode in range(1, 101):
        state = env.reset()
        total_reward = 0
        
        for t in range(3600/(GREEN_DURATION+YELLOW_DURATION)): #Number of updates
            
            action_idx = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action_idx)
            memory.push(state, action_idx, reward, next_state, done)
            loss = agent.train_step(memory, BATCH_SIZE, GAMMA, beta=0.4)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if step_count % TARGET_UPDATE == 0:
                agent.update_target_network()
                print(f"   [Update] Target Network Synced at Step {step_count}")
        
        epsilon = max(EPS_END, epsilon * 0.98)
        
        print(f"Episode {episode:03d} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}")

    print("Training Complete.")