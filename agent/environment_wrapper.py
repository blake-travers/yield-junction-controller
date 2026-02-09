import os
import torch
import numpy as np
import traci
import contextlib

from agent.vehicle import Vehicle

class SumoIntersectionEnv:
    """
    Enviroment Wrapper representing the SUMO environment and the connection between it and the trainable agent
    """
    def __init__(self, net_file, sumo_cmd, phases, lane_list, yellow_time, decision_frequency, num_cells, features, cell_length, device, reward_modifier):
        self.net_file = net_file
        self.sumo_cmd = sumo_cmd
        self.phases = phases
        self.lane_list = lane_list
        self.active_vehicles = {}
        self.tls_id = None
        
        self.yellow_time = yellow_time
        self.decision_frequency = decision_frequency
        self.num_cells = num_cells
        self.features = features
        self.cell_length = cell_length
        self.device = device
        self.reward_modifier = reward_modifier

    def reset(self):
        try: traci.close()
        except: pass

        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stderr(fnull), contextlib.redirect_stdout(fnull):
                traci.start(self.sumo_cmd)
                
        self.tls_id = traci.trafficlight.getIDList()[0]

        self.internal_lane_ids = [lane for lane in self.lane_list if ":" in lane] #Grab the Internal lanes
        controlled_links = traci.trafficlight.getControlledLinks(self.tls_id) #Grab the traci-linked lights
        self.lane_to_str_idx = {}

        for string_idx, links in enumerate(controlled_links): #For each controlled link
            for link in links:
                lane_id = link[2] #Get the id
                if lane_id in self.internal_lane_ids: #Map the id to the internal lane
                    self.lane_to_str_idx[lane_id] = string_idx


        self.active_vehicles = {}

        self.start_sim_time = traci.simulation.getTime()
        
        for _ in range(10):
            self._sim_step()
            
        return self.get_state()

    def step(self, action_idx):
        target_phase = self.phases[action_idx]
        current_phase = traci.trafficlight.getRedYellowGreenState(self.tls_id)

        self.throughput_this_step = 0
        
        if target_phase != current_phase: #If changing phases
            y_state = list(current_phase)
            for i in range(len(current_phase)):
                if current_phase[i].lower() == 'g' and target_phase[i].lower() == 'r': #If changing from green to red
                    y_state[i] = 'y' #Change to yellow first

            
            traci.trafficlight.setRedYellowGreenState(self.tls_id, "".join(y_state))
            
            for _ in range(self.yellow_time): #For YELLOW TIME
                self.throughput_this_step += self._sim_step()
            
            traci.trafficlight.setRedYellowGreenState(self.tls_id, target_phase) #Now switch all target g to green, and target r to red

            for _ in range(max(0, self.decision_frequency - self.yellow_time)): #For the rest of the duration we step the environment
                self.throughput_this_step += self._sim_step()
            
        else: #If staying the same, run the same for another 50 timesteps
            traci.trafficlight.setRedYellowGreenState(self.tls_id, target_phase)
            for _ in range(self.decision_frequency):
                self.throughput_this_step += self._sim_step()
            
        next_state = self.get_state()
        reward = self.get_reward()

        current_queue_len = len(self.active_vehicles)
        total_system_wait = sum([v.wait_time for v in self.active_vehicles.values()])
        avg_system_wait = total_system_wait / current_queue_len if current_queue_len > 0 else 0.0

        info = {
            "queue_len": current_queue_len,
            "avg_wait": avg_system_wait,
            "throughput": self.throughput_this_step
        }
        
        return next_state, reward, info

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
            lane_grid = np.zeros((self.num_cells, self.features), dtype=np.float32) #Initialise lane grid of size at the moment 50x3
            
            lane_cars = [v for v in self.active_vehicles.values() if v.getLaneID() == lane_id] #Get all vehicles in this current lane
            
            for car in lane_cars: #For each car in this lane
                position = car.getPosition() #Get the position of this car along the edge
                cell_idx = int(position / self.cell_length) #Get the closest index to this car position
                dir_vec = car.getDirection() #Get the direction of this car
                lane_grid[cell_idx] = dir_vec #Populate the index with this direction vector
            
            batch_features.append(lane_grid) #Flatten for MLP

        return torch.tensor(np.array([batch_features]), dtype=torch.float32).to(self.device)
    
    def get_phase_vector(self):
        """
        Returns a [1, 14] tensor representing the current Green/Red status 
        of the 14 internal lanes.
        """
        phase_str = traci.trafficlight.getRedYellowGreenState(self.tls_id) #Get the current phase string
        
        phase_vec = []
        for lane_id in self.internal_lane_ids: #For each internal lane id
            idx = self.lane_to_str_idx.get(lane_id) #Get the mapping
            if idx is not None and idx < len(phase_str):
                is_green = 1.0 if phase_str[idx].lower() == 'g' else 0.0  #If green population index with 1, else 0
            else:
                is_green = 0.0

            phase_vec.append(is_green)

        return torch.tensor([phase_vec], dtype=torch.float32).to(self.device)

    def get_reward(self):

        WAIT_COEFFICIENT = 0.00001 * 2 * self.decision_frequency #Normalised depending on frequency
        ACTIVE_COEFFICIENT = 0.001 * 13 * self.decision_frequency
        
        total_wait_penalty = -sum(traci.vehicle.getAccumulatedWaitingTime(vid)**1.5 for vid in self.active_vehicles)*WAIT_COEFFICIENT #With reward modifier of 0.2, this basically means -0.5 min and for 100 decisions thats -50
        active_vehicle_penalty = -len(self.active_vehicles)*ACTIVE_COEFFICIENT
        reward = self.throughput_this_step + active_vehicle_penalty# +  total_wait_penalty
        #print(f"Throughput Reward: {self.throughput_this_step:.0f} |  Active Vehicles Penalty: {active_vehicle_penalty:.2f} |  Waiting Penalty Total: {total_wait_penalty:.2f} |  Total Unmodified Reward: {reward:.2f} |  Total Modified Reward: {reward * self.reward_modifier:.3f}")
        return reward * self.reward_modifier
