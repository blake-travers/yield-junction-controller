import os
import torch
import numpy as np
import traci
import time
import random
import matplotlib.pyplot as plt

from agent.agent import DoubleDQNAgent, PrioritisedReplayBuffer
from agent.adjacency_matrix import get_adjacency_matrices
from agent.generate_phases import generate_rule_based_phases, create_phase_mask
from agent.environment_wrapper import SumoIntersectionEnv
from environment.generate_routes import generate_routes
from agent.modelGNN import TrafficGNN

SUMO_CFG = "environment/sim.sumocfg"
NET_FILE = "environment/basic_intersection.net.xml"
SUMO_CMD = ["sumo", "-c", SUMO_CFG, "--time-to-teleport", "-1", "--no-step-log", "--no-warnings", "--ignore-route-errors"]
SUMO_GUI_CMD = ["sumo-gui", "-c", SUMO_CFG, "--time-to-teleport", "-1", "--no-step-log", "--no-warnings", "--ignore-route-errors"]

BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.96
MEMORY_SIZE = 50000
REWARD_MODIFIER = 0.2
INITIAL_TAU = 0.015
FINAL_TAU = 0.001

EPISODE_LENGTH = 500
EPISODE_NUMBER = 200
EPISODE_PRINT_FREQUENCY = 5

DECISION_FREQUENCY = 25
YELLOW_TIME = 8
CELL_LENGTH = 1 #In Metres (basic intersection is of size 42 so cell size of 1 means 42 cells)
MAX_LANE_SIZE = 50
NUM_CELLS = int(MAX_LANE_SIZE // CELL_LENGTH)
FEATURES = 3 #Just Direction for now
SCOREABLE_LANES = 14
NUM_LANES = 26

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def aggregate_print_metrics(ep_metrics):
    avg_loss = np.mean(ep_metrics["loss"]) if ep_metrics["loss"] else 0.0
    reward = ep_metrics["reward"] / EPISODE_PRINT_FREQUENCY if ep_metrics["reward"] else 0.0
    avg_queue = np.mean(ep_metrics["queue_len"])
    avg_wait = np.mean(ep_metrics["wait_time"])
    avg_flush = ep_metrics["extra_timesteps"] / EPISODE_PRINT_FREQUENCY
    avg_td = np.mean(ep_metrics["td_error"]) if ep_metrics["td_error"] else 0
    avg_q = np.mean(ep_metrics["q_mean"]) if ep_metrics["q_mean"] else 0
    epsilon_mean = np.mean(ep_metrics["epsilon"]) if ep_metrics["epsilon"] else 0

    print("-" * 120)
    print(f"{f'Episodes: {(ep_metrics["episode"]+1-EPISODE_PRINT_FREQUENCY):03d}-{ep_metrics["episode"]:03d}':<{26}} |")

    print(f"{f'  Duration: {ep_metrics["duration"]:8.1f}s':<{26}} | "
        f"{f'Epsilon Mean: {epsilon_mean:7.3f}':<{21}} | "
        f"{f'Reward Mean: {reward:3.3f}':<{21}} | "
        f"{f'Average Loss: {avg_loss:9.4f}'}")

    print(f"{f'  Queue Length: {avg_queue:4.1f} cars':<{26}} | "
        f"{f'Time to Flush: {avg_flush:4.0f}':<{21}} | "
        f"{f'Average TD: {avg_td:3.3f}':<{21}} | "
        f"{f'Average Lane Q: {avg_q:7.4f}'}")

def train_agent():
    print("Loading Graph Adjacency matrices...")
    nodes, adj_flow, adj_conf = get_adjacency_matrices(NET_FILE)
    assert len(nodes) == NUM_LANES
    
    print("Generating valid Traffic light phases...")
    internal_indices = [i for i, node in enumerate(nodes) if ":" in node]
    internal_nodes = [nodes[i] for i in internal_indices]
    phases = generate_rule_based_phases(NET_FILE)
    phase_mask = create_phase_mask(NET_FILE, phases, internal_nodes)
    
    print("Setting up Intersection Environment...")
    env = SumoIntersectionEnv(NET_FILE, SUMO_CMD, phases, nodes, YELLOW_TIME, DECISION_FREQUENCY, NUM_CELLS, FEATURES, CELL_LENGTH, DEVICE, REWARD_MODIFIER)
    
    agent = DoubleDQNAgent(num_lanes=NUM_LANES, scoreable_lanes=SCOREABLE_LANES, num_phases=len(phases), input_dim=FEATURES, adj_flow=adj_flow, adj_conf=adj_conf, phase_mask=phase_mask, device=DEVICE)
    memory = PrioritisedReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPS_START
    training_history = []
    
    
    print(f"Starting Training on {DEVICE}. Tentative Maximum Reward: {(EPISODE_LENGTH*REWARD_MODIFIER)/4}")
    for print_episode in range(1, int(EPISODE_NUMBER/EPISODE_PRINT_FREQUENCY)):
        ep_metrics = {
            "reward": 0,
            "loss": [],
            "queue_len": [],
            "wait_time": [],
            "td_error": [],
            "q_mean": [],
            "throughput": 0,
            "action_counts": {},
            "epsilon": [],
            "episode": 0,
            "episode_start": 0,
            "extra_timesteps": 0
        }
        episode_start = time.time()
        for episode in range(0, EPISODE_PRINT_FREQUENCY):

            seed = random.randint(0, 1000000)
            frequency = max(2, min(8, random.gauss(4.75, 2.25)))
            generate_routes(seed, EPISODE_LENGTH, frequency)
            #print(f"{frequency:.2f}")

            state = env.reset()

            done = None
            current_phase = env.get_phase_vector() #Get initial phase
            current_sim_time = 0 #Set to zero for first while

            while True: #Go until no more active vehicles. As soon as "done" is true we break
                if current_sim_time >= EPISODE_LENGTH and len(env.active_vehicles) == 0 or current_sim_time >= EPISODE_LENGTH*10:
                    break
                current_sim_time = traci.simulation.getTime()
                action_idx = agent.select_action(state, current_phase, epsilon)
                next_state, reward, info = env.step(action_idx)
                next_phase = env.get_phase_vector() #Get next phase based upon this action
                memory.push(state, action_idx, reward, next_state, current_phase, next_phase, False if done is None else True)
                metrics = agent.train_step(memory, BATCH_SIZE, GAMMA, beta=0.6)
                agent.update_target_network(tau=max(FINAL_TAU, INITIAL_TAU * (epsilon ** 0.4))) #Update the target network slightly
                
                state = next_state
                current_phase = next_phase

                ep_metrics["reward"] += reward
                ep_metrics["throughput"] += info["throughput"]
                ep_metrics["queue_len"].append(info["queue_len"])
                ep_metrics["wait_time"].append(info["avg_wait"])
                ep_metrics["loss"].append(metrics["loss"])
                ep_metrics["td_error"].append(metrics["td_error"])
                ep_metrics["q_mean"].append(metrics["q_mean"])
                ep_metrics["action_counts"][int(action_idx)] = ep_metrics["action_counts"].get(int(action_idx), 0) + 1

            ep_metrics["extra_timesteps"] += current_sim_time - EPISODE_LENGTH
            ep_metrics["epsilon"].append(epsilon)
            epsilon = max(EPS_END, epsilon * EPS_DECAY)
            agent.scheduler.step()

        ep_metrics["episode"] = print_episode*EPISODE_PRINT_FREQUENCY
        ep_metrics["duration"] = time.time() - episode_start
        training_history.append(ep_metrics.copy())
        

        aggregate_print_metrics(ep_metrics)

    print("Training Complete. Saving Model...")
    torch.save(agent.policy_net.state_dict(), "traffic_gnn_model.pth")
    return training_history

def watch_agent():
    # 1. Re-generate the context needed for the model
    seed = random.randint(0, 1000000)
    frequency = max(2, min(8, random.gauss(4.75, 2.25)))
    generate_routes(seed, EPISODE_LENGTH*10, frequency)

    nodes, adj_flow, adj_conf = get_adjacency_matrices(NET_FILE)
    internal_indices = [i for i, node in enumerate(nodes) if ":" in node]
    internal_nodes = [nodes[i] for i in internal_indices]
    phases = generate_rule_based_phases(NET_FILE)
    phase_mask = create_phase_mask(NET_FILE, phases, internal_nodes).to(DEVICE)
    env = SumoIntersectionEnv(NET_FILE, SUMO_GUI_CMD, phases, nodes, YELLOW_TIME, DECISION_FREQUENCY, NUM_CELLS, FEATURES, CELL_LENGTH, DEVICE, REWARD_MODIFIER)

    model = TrafficGNN(input_dim=FEATURES, output_dim=SCOREABLE_LANES, num_lanes=NUM_LANES).to(DEVICE)
    model.load_state_dict(torch.load("traffic_gnn_model.pth"))
    model.eval() 

    print("Starting GUI Evaluation...")

    
    state = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            # Get lane priorities from the GNN
            lane_q = model(state, adj_flow.to(DEVICE), adj_conf.to(DEVICE), env.get_phase_vector())
            
            phase_q = torch.matmul(lane_q, phase_mask.t()) / SCOREABLE_LANES  #Pick the best phase
            action_idx = phase_q.argmax(dim=1).item()
            
        state, reward, info = env.step(action_idx)
        
        # Check if simulation time is up and intersection is clear
        current_sim_time = traci.simulation.getTime()
        if current_sim_time >= EPISODE_LENGTH*10 and traci.simulation.getMinExpectedNumber() == 0:
            done = True
            
    print(f"Evaluation Run Complete.")
    traci.close()

def plot_training_results(history):
    episodes = [h["episode"] for h in history]
    
    mean_rewards = [h["reward"] / EPISODE_PRINT_FREQUENCY for h in history]
    avg_q_values = [np.mean(h["q_mean"]) for h in history]
    avg_losses = [np.mean(h["loss"]) if h["loss"] else 0 for h in history]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle('Combined Traffic GNN Training Metrics')

    # 1. Reward (Left Axis)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Mean Reward', color='green')
    ax1.plot(episodes, mean_rewards, color='green', label='Mean Reward', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, alpha=0.3)

    # 2. Q-Value (Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Q-Value', color='blue')
    ax2.plot(episodes, avg_q_values, color='blue', label='Mean Q-Value', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='blue')

    # 3. Loss (Offset Right Axis)
    ax3 = ax1.twinx()
    # Offset the third axis so it doesn't sit on top of the Q-value axis
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Average Loss', color='red')
    ax3.plot(episodes, avg_losses, color='red', label='Average Loss', alpha=0.6)
    ax3.tick_params(axis='y', labelcolor='red')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.tight_layout()
    plt.savefig("combined_metrics.png")
    plt.show()

if __name__ == "__main__":
    #history = train_agent()
    #plot_training_results(history)
    watch_agent()
