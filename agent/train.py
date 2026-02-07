import os
import torch
import numpy as np
import traci
import time
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

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
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 0.83
MEMORY_SIZE = 50000
REWARD_MODIFIER = 0.2
INITIAL_TAU = 0.03
FINAL_TAU = 0.001

EPISODE_LENGTH = 500
EPISODE_NUMBER = 300
EPISODE_PRINT_FREQUENCY = 5

DECISION_FREQUENCY = 5
YELLOW_TIME = 4
MIN_GREEN = 15 #Minimum amount of time on green light
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
    env = SumoIntersectionEnv(NET_FILE, SUMO_CMD, phases, nodes, YELLOW_TIME, DECISION_FREQUENCY, NUM_CELLS, FEATURES, CELL_LENGTH, DEVICE, REWARD_MODIFIER, MIN_GREEN)
    
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
            frequency = max(1, min(5, random.gauss(2.75, 2.25)))
            generate_routes(seed, EPISODE_LENGTH, frequency)
            #print(f"{frequency:.2f}")

            state = env.reset()

            done = None
            current_phase = env.get_phase_vector() #Get initial phase
            current_sim_time = 0 #Set to zero for first while

            while True: #Go until no more active vehicles. As soon as "done" is true we break
                if current_sim_time >= EPISODE_LENGTH and len(env.active_vehicles) == 0 or current_sim_time >= EPISODE_LENGTH*3:
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

    # 2. Q-Value AND Loss (Shared Right Axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Q-Value & Loss', color='black') # Label covers both
    
    # Plot Q-Values (Blue dashed)
    ax2.plot(episodes, avg_q_values, color='blue', label='Mean Q-Value', linestyle='--')
    
    # Plot Loss (Red solid) - ON THE SAME AXIS (ax2)
    ax2.plot(episodes, avg_losses, color='red', label='Average Loss', alpha=0.6)
    
    ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends (Only need ax1 and ax2 now)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig("combined_metrics.png")
    plt.show()

def run_gnn_simulation(gui=False, episode_length=EPISODE_LENGTH):
    """
    Runs an evaluation example for the trained GNN agent based on the existing routes xml file
    Returns detailed episode metrics
    """
    nodes, adj_flow, adj_conf = get_adjacency_matrices(NET_FILE)
    internal_indices = [i for i, node in enumerate(nodes) if ":" in node]
    internal_nodes = [nodes[i] for i in internal_indices]

    phases = generate_rule_based_phases(NET_FILE)
    phase_mask = create_phase_mask(NET_FILE, phases, internal_nodes).to(DEVICE)
    env = SumoIntersectionEnv(NET_FILE, SUMO_GUI_CMD if gui else SUMO_CMD, phases, nodes, YELLOW_TIME, DECISION_FREQUENCY, NUM_CELLS, FEATURES, CELL_LENGTH, DEVICE, REWARD_MODIFIER)

    model = TrafficGNN(input_dim=FEATURES, output_dim=SCOREABLE_LANES, num_lanes=NUM_LANES).to(DEVICE)
    model.load_state_dict(torch.load("traffic_gnn_model.pth"))
    model.eval() 

    print(f"Starting GNN Evaluation. GUI mode: {"On" if gui else "Off"}.")

    state = env.reset()
    done = False
    total_reward = 0
    speeds = []
    arrived_vehicles = 0
    queue_lengths = []
    time_losses = []
    max_wait_time_observed = 0.0
    speeds = []
    track_waits = {}

    while not done:
        with torch.no_grad():
            lane_q = model(state, adj_flow.to(DEVICE), adj_conf.to(DEVICE), env.get_phase_vector())
            phase_q = torch.matmul(lane_q, phase_mask.t()) / SCOREABLE_LANES
            action_idx = phase_q.argmax(dim=1).item()

        state, reward, info = env.step(action_idx)
        total_reward += reward

        veh_ids = traci.vehicle.getIDList()

        if len(veh_ids) > 0:
            queue_lengths.append(sum(1 for vid in veh_ids if traci.vehicle.getSpeed(vid) < 0.1)) #Track Queue Lengths
            speeds.extend([traci.vehicle.getSpeed(vid) for vid in veh_ids]) #Track Current Speeds
            time_losses.extend([traci.vehicle.getTimeLoss(vid) for vid in veh_ids]) #Track Time lost

            current_waits = [traci.vehicle.getAccumulatedWaitingTime(vid) for vid in veh_ids] #Track the longest car waiting time and average waiting time

            for i, vid in enumerate(veh_ids):
                track_waits[vid] = current_waits[i]
            
            if current_waits:
                step_max_wait = max(current_waits)
                if step_max_wait > max_wait_time_observed:
                    max_wait_time_observed = step_max_wait

        arrived_vehicles += traci.simulation.getArrivedNumber()

        current_sim_time = traci.simulation.getTime()
        if current_sim_time >= episode_length: #Terminate regardless of if empty or not
            done = True

    traci.close()

    avg_speed = (np.mean(speeds) * 3.6) if speeds else 0.0 # Convert m/s to km/h
    avg_queue = np.mean(queue_lengths) if queue_lengths else 0.0
    avg_time_loss = np.mean(time_losses) if time_losses else 0.0
    avg_wait = np.mean(list(track_waits.values())) if track_waits else 0.
    throughput_rate = (arrived_vehicles / episode_length) * 3600 # Vehicles per hour

    return {
        "Avg Wait Time (s)": round(avg_wait, 2),
        "Avg Queue Length (veh)": round(avg_queue, 2),
        "Avg Speed (km/h)": round(avg_speed, 2),
        "Avg Time Loss (s)": round(avg_time_loss, 2),
        "Max Wait Time (s)": round(max_wait_time_observed, 2),
        "Throughput (veh/hr)": round(throughput_rate, 2),
        "Total Reward": round(total_reward, 2)
    }

def run_baseline_simulation(gui=False, episode_length=500):
    """
    Heuristic Baseline: "Oracle Lane Scoring"
    
    1. Identifies the 14 Internal Lanes (e.g., :J1_0_0) that the Phase Mask controls.
    2. Counts exactly how many vehicles want to enter each of those 14 lanes.
    3. Constructs a [1, 14] Score Vector.
    4. Multiplies by the [14, 848] Phase Mask to find the best Phase.
    """
    
    # 1. SETUP: Generate the exact same Matrices and Masks as the GNN
    nodes, _, _ = get_adjacency_matrices(NET_FILE) # 'nodes' is length 26
    
    # Extract ONLY the 14 Internal Nodes (this matches the Phase Mask dimensions)
    internal_indices = [i for i, node in enumerate(nodes) if ":" in node]
    internal_nodes = [nodes[i] for i in internal_indices] # Length 14
    
    phases = generate_rule_based_phases(NET_FILE)
    # The Mask shape is [Num_Phases, Num_Internal_Lanes] -> [848, 14]
    phase_mask = create_phase_mask(NET_FILE, phases, internal_nodes).to(DEVICE)
    
    cmd_mode = SUMO_GUI_CMD if gui else SUMO_CMD
    env = SumoIntersectionEnv(NET_FILE, cmd_mode, phases, nodes, YELLOW_TIME, DECISION_FREQUENCY, NUM_CELLS, FEATURES, CELL_LENGTH, DEVICE, REWARD_MODIFIER)
    
    print(f"Starting Heuristic Baseline (Oracle Lane Scoring). GUI: {gui}")

    # Metrics
    total_reward = 0
    waiting_times = []       
    queue_lengths = []
    time_losses = []
    speeds = []
    arrived_vehicles = 0
    max_wait_time_observed = 0.0
    track_waits = {} 

    state = env.reset()
    done = False
    
    # --- MAP: Signal Index -> Internal Lane ID ---
    # We need this to know which internal lane a car is targeting
    tls_id = traci.trafficlight.getIDList()[0]
    links = traci.trafficlight.getControlledLinks(tls_id)
    signal_to_internal = {}
    for idx, connections in enumerate(links):
        if len(connections) > 0:
            # connection = (Incoming, Outgoing, Via)
            # We map Signal Index -> Via Lane (Internal)
            signal_to_internal[idx] = connections[0][2]

    while not done:
        
        # --- 1. CALCULATE SCORES FOR INTERNAL LANES ---
        # We need to build a score for each of the 14 internal_nodes
        internal_lane_counts = {lane: 0 for lane in internal_nodes}

        # [FRESH LIST] Get IDs for Decision Making
        decision_veh_ids = traci.vehicle.getIDList()
        
        for vid in decision_veh_ids:
            # getNextTLS returns: [(tlsID, tlsIndex, dist, state), ...]
            next_tls = traci.vehicle.getNextTLS(vid)
            
            for tls_info in next_tls:
                t_id, t_index, t_dist, _ = tls_info
                if t_id == tls_id:
                    # Found the target signal index. Map to Internal Lane ID.
                    if t_index in signal_to_internal:
                        target_lane = signal_to_internal[t_index]
                        
                        # Increment score for this internal lane
                        if target_lane in internal_lane_counts:
                            internal_lane_counts[target_lane] += 1
                    break # Only count the immediate next light

        # --- 2. CONSTRUCT SCORE TENSOR [1, 14] ---
        # We must iterate internal_nodes to ensure the order matches phase_mask
        ordered_scores = [internal_lane_counts[lane] for lane in internal_nodes]
        lane_score_tensor = torch.tensor([ordered_scores], dtype=torch.float32).to(DEVICE)
        
        # --- 3. CALCULATE PHASE SCORES ---
        # [1, 14] @ [14, 848] = [1, 848]
        # We transpose phase_mask from [848, 14] to [14, 848]
        phase_scores = torch.matmul(lane_score_tensor, phase_mask.t())
        
        # --- 4. SELECT ACTION ---
        action_idx = phase_scores.argmax(dim=1).item()
        
        # --- 5. STEP ENVIRONMENT ---
        # This advances the simulation by 25 seconds! Old vehicle lists are now stale.
        next_state, reward, info = env.step(action_idx)
        total_reward += reward
        
        # --- 6. COLLECT METRICS ---
        # [FIX] Refresh the list of vehicles immediately after the step
        # This prevents "Vehicle not known" errors for cars that left during the step
        current_veh_ids = traci.vehicle.getIDList()
        
        if len(current_veh_ids) > 0:
            halting_count = sum(1 for vid in current_veh_ids if traci.vehicle.getSpeed(vid) < 0.1)
            queue_lengths.append(halting_count)

            current_waits = []
            for vid in current_veh_ids:
                w_time = traci.vehicle.getAccumulatedWaitingTime(vid)
                current_waits.append(w_time)
                track_waits[vid] = w_time 
                speeds.append(traci.vehicle.getSpeed(vid))
                time_losses.append(traci.vehicle.getTimeLoss(vid))

            if current_waits:
                step_max = max(current_waits)
                if step_max > max_wait_time_observed:
                    max_wait_time_observed = step_max

        arrived_vehicles += traci.simulation.getArrivedNumber()
        
        if traci.simulation.getTime() >= episode_length:
            done = True
            
    traci.close()
    
    # Final Metrics Calculation
    avg_wait = np.mean(list(track_waits.values())) if track_waits else 0.0
    avg_speed = (np.mean(speeds) * 3.6) if speeds else 0.0 
    avg_queue = np.mean(queue_lengths) if queue_lengths else 0.0
    avg_time_loss = np.mean(time_losses) if time_losses else 0.0
    throughput_rate = (arrived_vehicles / episode_length) * 3600

    return {
        "Avg Wait Time (s)": round(avg_wait, 2),
        "Avg Queue Length (veh)": round(avg_queue, 2),
        "Avg Speed (km/h)": round(avg_speed, 2),
        "Avg Time Loss (s)": round(avg_time_loss, 2),
        "Max Wait Time (s)": round(max_wait_time_observed, 2),
        "Throughput (veh/hr)": round(throughput_rate, 2),
        "Total Reward": round(total_reward, 2)
    }

def watch_agent():
    """
    Generates a Route to watch the GNN perform
    """
    seed = random.randint(0, 1000000)
    frequency = 3 #Constant frequency for now... should do multiple different ones for baseline check
    print(f"Generating routes with freq {frequency:.2f}...")
    generate_routes(seed, EPISODE_LENGTH*10, frequency) # Long episode for watching

    metrics = run_gnn_simulation(gui=True, episode_length=EPISODE_LENGTH)
    print(f"Watch Session Metrics: {metrics}")

def compare_agents():
    """
    Compares a GNN and Baseline Heuristic implementation, and prints a comparison table for the report
    """
    seed = random.randint(0, 1000000)
    frequency = 3
    print("Generating standard validation traffic...")
    generate_routes(seed, EPISODE_LENGTH, frequency)

    gnn_metrics = run_gnn_simulation(gui=False, episode_length=EPISODE_LENGTH)
    base_metrics = run_baseline_simulation(gui=True, episode_length=EPISODE_LENGTH)

    headers = ["Metric", "Baseline", "TrafficGNN", "Diff"]
    table_data = []

    for key in gnn_metrics.keys(): #Calculate data
        base_val = base_metrics.get(key, 0)
        gnn_val = gnn_metrics.get(key, 0)
        
        if base_val != 0:
            diff = ((gnn_val - base_val) / base_val) * 100
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "N/A"
            
        table_data.append([key, base_val, gnn_val, diff_str])

    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(tabulate(table_data, headers=headers, tablefmt="grid")) #Print table in nice format

if __name__ == "__main__":
    #history = train_agent()
    #plot_training_results(history)
    watch_agent()
    #compare_agents()
