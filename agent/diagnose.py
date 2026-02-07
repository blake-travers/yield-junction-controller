import traci
import os
import sys
import numpy as np
from agent.adjacency_matrix import get_adjacency_matrices
from agent.generate_phases import generate_rule_based_phases

# Import your constants
from agent.train import NET_FILE, SUMO_CMD, SUMO_GUI_CMD

def debug_tls_mapping():
    print("--- STARTING DEBUG DIAGNOSTIC ---")
    
    # 1. Start SUMO in CLI mode (fast)
    # We need to start it to query the API
    traci.start(SUMO_CMD)
    
    # 2. Get the Intersection ID
    tls_ids = traci.trafficlight.getIDList()
    if not tls_ids:
        print("ERROR: No Traffic Lights found in the simulation!")
        traci.close()
        return

    tls_id = tls_ids[0]
    print(f"Target Traffic Light ID: {tls_id}")

    # 3. Get the Controlled Links
    # This tells us: Index 0 -> controls Lane X
    # Structure: links[index] = [(incoming, outgoing, via), ...]
    links = traci.trafficlight.getControlledLinks(tls_id)
    
    print(f"\n[MAPPING DIAGNOSIS]")
    print(f"The Traffic Light string has {len(links)} characters.")
    print(f"{'Index':<6} | {'Status':<10} | {'Incoming Lane ID (The one we check queue on)'}")
    print("-" * 60)

    # We also get the current state string to see what's green/red right now
    current_state = traci.trafficlight.getRedYellowGreenState(tls_id)

    valid_lane_ids = []

    for i, connections in enumerate(links):
        # connections is a list of tuples: [(in, out, via), ...]
        if not connections:
            print(f"{i:<6} | {current_state[i]:<10} | NO CONNECTIONS (Dummy Signal?)")
            continue
            
        # We only care about the FIRST connection for the queue length
        # (Usually all connections at this index share the same incoming lane)
        incoming_lane = connections[0][0] 
        
        print(f"{i:<6} | {current_state[i]:<10} | {incoming_lane}")
        valid_lane_ids.append(incoming_lane)

    # 4. Check Phases Object
    print(f"\n[PHASE CHECK]")
    phases = generate_rule_based_phases(NET_FILE)
    print(f"Generated {len(phases)} phases.")
    
    # Let's inspect Phase 0
    p0 = phases[0]
    # Handle if p0 is an object or string
    p0_str = p0.state if hasattr(p0, 'state') else p0
    
    print(f"Phase 0 String: {p0_str}")
    print(f"Length match? {'YES' if len(p0_str) == len(links) else 'NO (CRITICAL ERROR)'}")

    # 5. Simulate a Score Calculation
    print(f"\n[SCORING SIMULATION]")
    print("Testing if we can read queue lengths for Phase 0...")
    
    active_lanes = set()
    for char_idx, char in enumerate(p0_str):
        if char in ['G', 'g']:
            if char_idx < len(links) and links[char_idx]:
                lane = links[char_idx][0][0]
                active_lanes.add(lane)
    
    print(f"Phase 0 would check queues on lanes: {active_lanes}")
    
    # Validate these lanes exist
    for lane in active_lanes:
        try:
            q = traci.lane.getLastStepHaltingNumber(lane)
            print(f"  -> Lane '{lane}' read successfully. Queue: {q}")
        except Exception as e:
            print(f"  -> ERROR Reading Lane '{lane}': {e}")

    traci.close()
    print("\n--- DIAGNOSTIC COMPLETE ---")

if __name__ == "__main__":
    debug_tls_mapping()