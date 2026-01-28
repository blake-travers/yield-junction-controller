import sumolib
import numpy as np
import itertools
from adjacency_matrix import get_adjacency_matrices

#In a merge / give way, Straight has right of way, followed by LEFT and the RIGHT
#Based upon AUS / UK laws, RIGHT TURNS can only intersect with STRAIGHTS under one of the two possible conditions
LEFT_TURNS =  {0, 4, 7, 11} #Left turns never intersect with each other
STRAIGHTS =   {1, 2, 8, 9, 5, 12} #Intersecting Straight turns CANNOT be in the same phase
RIGHT_TURNS = {3, 6, 10, 13} #Intersecting Right turns CANNOT be in the same phase

def generate_rule_based_phases(net_path):
    print("Generating adjacency matrices...")
    lane_ids, _, lane_conf_matrix = get_adjacency_matrices(net_path) #Get the conflict matrix - this is important to determine which lanes intersect
    print("Generating all possible rule-based phases...")
    lane_to_idx = {l: i for i, l in enumerate(lane_ids)}

    net = sumolib.net.readNet(net_path, withInternal=True) #Grab the network
    tls = net.getTrafficLights()[0] #Grab the traffic light

    signal_to_lanes = {i: [] for i in range(20)} # Pre-allocate enough slots (safe upper bound)
    real_max_index = 0

    # tls.getConnections() returns a list of [SourceLane, TargetLane, LinkIndex]
    for conn in tls.getConnections(): 
        source_lane = conn[0]
        target_lane = conn[1]
        link_index = conn[2] # The signal index (0, 1, 2...)
        
        real_max_index = max(real_max_index, link_index)

        # To get the Internal Lane (:J_x_y), we must look at the source lane's outgoing connections
        for outgoing in source_lane.getOutgoing():
            # Find the specific connection that goes to our target
            if outgoing.getToLane().getID() == target_lane.getID():
                via_id = outgoing.getViaLaneID()
                if via_id:
                    signal_to_lanes[link_index].append(via_id)

    # Clean up empty indices if any
    signal_to_lanes = {k: v for k, v in signal_to_lanes.items() if k <= real_max_index}
    num_signals = real_max_index + 1
    # --- FIX END ---

    print(f"Detected {num_signals} signals.")

    all_candidates = list(itertools.product([0, 1], repeat=num_signals)) #Generate all possible combinations of phases, ready to be pruned
    valid_phase_strings = [] #We now need to prune

    for phase_vector in all_candidates: #For each candidate
    
        active_indices = [i for i, val in enumerate(phase_vector) if val == 1] #Get the indicies of lights that are active at this point
        
        is_safe = True #Assume safe
        must_yield = set() 

        for idx1 in active_indices:
            for idx2 in active_indices: #Iterate through index pairs of these active lights
                if idx1 == idx2: continue
                
                max_conflict = 0.0
                lanes1 = signal_to_lanes[idx1]
                lanes2 = signal_to_lanes[idx2]
                
                for l1 in lanes1: #Check amount of conflict these two lanes have
                    for l2 in lanes2:
                        if l1 in lane_to_idx and l2 in lane_to_idx: #This should always be true
                            c = lane_conf_matrix[lane_to_idx[l1], lane_to_idx[l2]].item()
                            if c > max_conflict: max_conflict = c
                
                if max_conflict > 0.9: #If Intersecting, there is only one very specific case in which we can allow this phase
                    if (idx1 in RIGHT_TURNS and idx2 in STRAIGHTS): #And that is when we are turning right when the other guys is turning Left (TODO Prune this step further for AUS road rules!)
                        must_yield.add(idx1) #Add this right trurn into one of the yield conditions
                    elif (idx1 in STRAIGHTS and idx2 in RIGHT_TURNS): #Dont care about other way around because that is already considered in the original pair loop
                         pass 
                    else: #Any other case, ban it
                        is_safe = False
                        break

                elif max_conflict > 0.4: #If we need to merge instead, add left turns to yields
                    if idx1 in LEFT_TURNS and idx2 in STRAIGHTS:
                        must_yield.add(idx1)

            if not is_safe: break
        
        if is_safe: #Once the for loop is done, we convert this entire thing into a string
            chars = ["r"] * num_signals #Assume all red
            for idx in active_indices:
                if idx in must_yield:
                    chars[idx] = "g" #Add permissive green when appropriate
                else:
                    chars[idx] = "G" #Add proper green when appropriate
            
            valid_phase_strings.append("".join(chars))

    unique_phases = sorted(list(set(valid_phase_strings))) #Remove exact copies from the final list
    
    print(f"Complete. Found {len(unique_phases)} valid LHD phases.")
    return unique_phases
        

if __name__ == "__main__":
    phases = generate_rule_based_phases("environment/basic_intersection.net.xml")
    for i, p in enumerate(phases[:1000]):
        if i % 100 == 0:
            print(f"{i}: {p}")