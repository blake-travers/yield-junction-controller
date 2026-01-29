import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from agent.modelGNN import TrafficGNN

class PrioritizedReplayBuffer:
    """
    PER works by calculating the most important replays to train on first. Edge cases will therefore be better represented and dealt with in the model
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity #Size of memory bank
        self.alpha = alpha #Prioritisation of each element in the memory bank
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32) #List of priorities capacity long
        self.pos = 0 #We use position instead of queue becuase it is lower complexity O(1) instead of O(n)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of memories, based upon how important they are (alpha) and ensuring not to overdo them (beta)
        beta: How much the learning rate is reduced, as we don't want to overtrain to specific examples with high priority too much
        """
        if len(self.buffer) == 0:
            return None, None, None

        if len(self.buffer) == len(self.priorities):
            probs = self.priorities
        else:
            probs = self.priorities[:len(self.buffer)]
            
        probs = probs ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs) #Sample the index of each memory, weighted randomly and based upon alpha
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta) #Change the learning rate weights
        weights /= weights.max()
        
        batch = list(zip(*samples))
        
        return batch, indices, weights #Return the batch representin the list of memories and the weight applied to them

    def update_priorities(self, indices, new_priorities):
        for idx, prio in zip(indices, new_priorities): #Updates the priority value of each memory after computing Temporal Difference Loss
            self.priorities[idx] = prio

class DoubleDQNAgent:
    def __init__(self, num_lanes, num_phases, input_dim, adj_flow, adj_conf, phase_mask, device):
        """
        phase_mask: Bool Tensor [Num_Phases, Num_Lanes]. 1 if lane is Green in phase.
        adj_flow/conf: The adjacency matrices for the GNN.
        """
        self.device = device
        self.num_lanes = num_lanes
        self.num_phases = num_phases
        self.phase_mask = phase_mask.to(device)
        
        self.adj_flow = adj_flow.to(device)
        self.adj_conf = adj_conf.to(device)

        self.policy_net = TrafficGNN(input_dim, hidden_dim=64).to(device) #Create Policy and Target net
        self.target_net = TrafficGNN(input_dim, hidden_dim=64).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict()) #Make the target net identical to the policy first time
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss(reduction='none')

    def _get_phase_q_values(self, lane_q_values): #Sum up the lane q values to get the highest phase
        phase_q_values = torch.matmul(lane_q_values, self.phase_mask.t())
        return phase_q_values

    def select_action(self, state, epsilon):
        """
        Select an phase based upon the current state and the policy net
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_phases - 1)
        
        with torch.no_grad():
            state = state.to(self.device)
            lane_q = self.policy_net(state, self.adj_flow, self.adj_conf)
            phase_q = self._get_phase_q_values(lane_q)
            return phase_q.argmax(dim=1).item()

    def train_step(self, buffer, batch_size, gamma, beta):

        batch, indices, weights = buffer.sample(batch_size, beta) #Sample a PER batch
        if batch is None: return 0.0 #Happens if we don't have any experiences yet
        
        states, actions, rewards, next_states, dones = batch
        
        #Prepare all the information from this batch for training
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device).unsqueeze(1)

        current_lane_q = self.policy_net(states, self.adj_flow, self.adj_conf)
        current_phase_q_all = self._get_phase_q_values(current_lane_q) #Get the all the Q phase values represented by the policy net
        current_q = current_phase_q_all.gather(1, actions)

        with torch.no_grad():
            next_lane_q_policy = self.policy_net(next_states, self.adj_flow, self.adj_conf)
            next_phase_q_policy = self._get_phase_q_values(next_lane_q_policy) #Figure out the possible phase q values for each of the next states
            best_next_actions = next_phase_q_policy.argmax(dim=1).unsqueeze(1) #Figure out the best actions for the next states using the policy net
            
            next_lane_q_target = self.target_net(next_states, self.adj_flow, self.adj_conf)
            next_phase_q_target = self._get_phase_q_values(next_lane_q_target) #Determine the target phase based upon the target net
            next_q_value = next_phase_q_target.gather(1, best_next_actions) #Figure out the q value from the previous policy actions
            
            target_q = rewards + (gamma * next_q_value * (1 - dones)) #Compare the two and the rewards

        td_error = torch.abs(current_q - target_q) #Calculate Temporal Difference error - heart of DQN
        loss = (self.loss_fn(current_q, target_q) * weights).mean()

        self.optimizer.zero_grad() #Determine loss and Backpropagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) #Clip like in PPO
        self.optimizer.step()

        new_priors = td_error.detach().cpu().numpy().flatten() + 1e-6
        buffer.update_priorities(indices, new_priors)

        return loss.item()

    def update_target_network(self): #Iterativelyt updates the target network when appropriate. Target network aims to prevent training instability
        self.target_net.load_state_dict(self.policy_net.state_dict())