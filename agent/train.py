import os
import sys
import torch
import numpy as np
import traci
import sumolib
from collections import deque
import time

#TODO: Cars turning right currently spawn in the left lane, need to fix routing

from agent.agent import DoubleDQNAgent, PrioritisedReplayBuffer
from agent.vehicle import Vehicle
from agent.adjacency_matrix import get_adjacency_matrices
from agent.generate_phases import generate_rule_based_phases, create_phase_mask

SUMO_CFG = "environment/sim.sumocfg"
NET_FILE = "environment/basic_intersection.net.xml"
SUMO_CMD = ["sumo", "-c", SUMO_CFG, "--time-to-teleport", "-1", "--no-step-log", "--no-warnings"] # "sumo-gui" to watch / "sumo" to train

BATCH_SIZE = 64
GAMMA = 0.98
EPISODE_LENGTH = 3600 #TODO Prevent hardcode
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.97
MEMORY_SIZE = 50000
REWARD_MODIFIER = 0.01

GREEN_DURATION = 35
YELLOW_DURATION = 15
CELL_LENGTH = 1
MAX_LANE_SIZE = 50
NUM_CELLS = int(MAX_LANE_SIZE // CELL_LENGTH)
FEATURES = 3 #Just Direction for now

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SumoIntersectionEnv:
    """
    Enviroment Wrapper representing the SUMO environment and the connection between it and the trainable agent
    """
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
        
        self.throughput_this_step = 0
        
        if target_phase != current_phase:
            y_state = list(current_phase)
            for i in range(len(current_phase)):
                if current_phase[i].lower() == 'g' and target_phase[i].lower() == 'r':
                    y_state[i] = 'y'
            y_str = "".join(y_state)
            
            traci.trafficlight.setRedYellowGreenState(self.tls_id, y_str)
            for _ in range(YELLOW_DURATION):
                # Accumulate finished cars during Yellow
                self.throughput_this_step += self._sim_step()
        
        traci.trafficlight.setRedYellowGreenState(self.tls_id, target_phase)
        
        for _ in range(GREEN_DURATION):
            # Accumulate finished cars during Green
            self.throughput_this_step += self._sim_step()
            
        next_state = self.get_state()
        reward = self.get_reward()
        done = False

        current_queue_len = len(self.active_vehicles)
        total_system_wait = sum([v.wait_time for v in self.active_vehicles.values()])
        avg_system_wait = total_system_wait / current_queue_len if current_queue_len > 0 else 0.0

        info = {
            "queue_len": current_queue_len,
            "avg_wait": avg_system_wait,
            "throughput": self.throughput_this_step
        }
        
        return next_state, reward, done, info

    def _sim_step(self):
        traci.simulationStep()
        current_ids = set(traci.vehicle.getIDList())
        
        for vid in current_ids: #For each vehicle
            if vid not in self.active_vehicles: #If a new vehicle
                self.active_vehicles[vid] = Vehicle(vid) #Create a new class
            self.active_vehicles[vid].update() #Update all vehicles including this one
            
        departed = set(self.active_vehicles.keys()) - current_ids #Get list of done vehicles
        departed_count = len(departed)
        for vid in departed:
            del self.active_vehicles[vid] #Delete this done vehicles

        return departed_count #Return for reward

    def get_state(self):
        """
        Builds the Discretized Lane Grid.
        Shape: [1, Num_Lanes, 48]
        """
        batch_features = []

        for lane_id in self.lane_list:
            lane_grid = np.zeros((NUM_CELLS, FEATURES), dtype=np.float32) #Initialise lane grid of size at the moment 50x3
            
            lane_cars = [v for v in self.active_vehicles.values() if v.getLaneID() == lane_id] #Get all vehicles in this current lane
            
            for car in lane_cars: #For each car in this lane
                position = car.getPosition() #Get the position of this car along the edge
                cell_idx = int(position / CELL_LENGTH) #Get the closest index to this car position
                dir_vec = car.getDirection() #Get the direction of this car
                lane_grid[cell_idx] = dir_vec #Populate the index with this direction vector
            
            batch_features.append(lane_grid.flatten()) #Flatten for MLP

        return torch.tensor(np.array([batch_features]), dtype=torch.float32).to(DEVICE)

    def get_reward(self):
        """
        Reward for now is just the cars that make it through.
        """
        return float(self.throughput_this_step * REWARD_MODIFIER)
    

if __name__ == "__main__":
    print("Loading Graph Aadjacency matrices...")
    nodes, adj_flow, adj_conf = get_adjacency_matrices(NET_FILE)
    
    print("Generating valid Traffic light phases...")
    phases = generate_rule_based_phases(NET_FILE)
    phase_mask = create_phase_mask(NET_FILE, phases, nodes)
    
    print("Setting up Intersection Environment...")
    env = SumoIntersectionEnv(NET_FILE, SUMO_CMD, phases, nodes)
    
    agent = DoubleDQNAgent(num_lanes=len(nodes), num_phases=len(phases), input_dim=NUM_CELLS * FEATURES, adj_flow=adj_flow, adj_conf=adj_conf, phase_mask=phase_mask, device=DEVICE)
    memory = PrioritisedReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPS_START
    step_count = 0
    
    print(f"Starting Training on {DEVICE}...")
    
    for episode in range(1, 100):
        state = env.reset()
        episode_start = time.time()

        ep_metrics = {
            "reward": 0,
            "loss": [],
            "queue_len": [],
            "wait_time": [],
            "td_error": [],
            "q_mean": [],
            "throughput": 0,
            "action_counts": {}
        }
        
        for t in range(EPISODE_LENGTH//(GREEN_DURATION+YELLOW_DURATION)): #Number of updates
            
            action_idx = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action_idx)
            memory.push(state, action_idx, reward, next_state, done)
            metrics = agent.train_step(memory, BATCH_SIZE, GAMMA, beta=0.4)
            agent.update_target_network(tau=0.005) #Update the target network slightly
            
            state = next_state

            ep_metrics["reward"] += reward
            ep_metrics["throughput"] += info["throughput"]
            ep_metrics["queue_len"].append(info["queue_len"])
            ep_metrics["wait_time"].append(info["avg_wait"])
            if metrics is not None:
                ep_metrics["loss"].append(metrics["loss"])
                ep_metrics["td_error"].append(metrics["td_error"])
                ep_metrics["q_mean"].append(metrics["q_mean"])
            ep_metrics["action_counts"][int(action_idx)] = ep_metrics["action_counts"].get(int(action_idx), 0) + 1

        
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        avg_loss = np.mean(ep_metrics["loss"]) if ep_metrics["loss"] else 0.0
        avg_queue = np.mean(ep_metrics["queue_len"])
        avg_wait = np.mean(ep_metrics["wait_time"])
        duration = time.time() - episode_start
        avg_td = np.mean(ep_metrics["td_error"]) if ep_metrics["td_error"] else 0
        avg_q = np.mean(ep_metrics["q_mean"]) if ep_metrics["q_mean"] else 0

        print("-" * 60)
        print(f"Episode {episode:02d}")
        print(f"  Time: {duration:.0f}s")
        print(f"  Epsilon: {epsilon:.3f}")
        print(f"  Reward: {ep_metrics['reward']:.1f}")
        print(f"  Average Queue length: {avg_queue:.1f} cars")
        print(f"  Average Wait time: {avg_wait:.1f}s")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Average TD Error: {avg_td:.4f}")
        print(f"  Average Lane Q-value: {avg_q:.4f}")
        #print(f"  Actions: {dict(sorted(ep_metrics['action_counts'].items()))}")
        

    print("Training Complete.")