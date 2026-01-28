import os
import sys
import sumolib
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def get_full_internal_shape(lane_obj, net):
    """
    Stitches shapes based on Topology first, then Geometry fallback.
    """
    raw_shape = lane_obj.getShape()
    full_coords = list(raw_shape)
    
    curr_lane = lane_obj
    
    # Pre-fetch all internal lanes for geometric lookup (optimization)
    # In a real run, pass this map in to avoid rebuilding it every time
    if not hasattr(get_full_internal_shape, "start_map"):
        get_full_internal_shape.start_map = {}
        for edge in net.getEdges(withInternal=True):
            for l in edge.getLanes():
                if l.getID().startswith(":"):
                    start_coord = l.getShape()[0]
                    # Map (x, y) -> lane_object
                    # Rounding to avoid float precision misses
                    key = (round(start_coord[0], 2), round(start_coord[1], 2))
                    get_full_internal_shape.start_map[key] = l

    # Follow the chain
    for _ in range(10):
        next_internal = None
        
        # Method A: Try Topology (Standard)
        outgoing = curr_lane.getOutgoing()
        for conn in outgoing:
            target = conn.getToLane()
            if target.getID().startswith(":"):
                next_internal = target
                break
        
        # Method B: Try Geometry (Fallback)
        if not next_internal:
            last_point = full_coords[-1]
            key = (round(last_point[0], 2), round(last_point[1], 2))
            
            # Check if any internal lane starts exactly here
            candidate = get_full_internal_shape.start_map.get(key)
            
            # Ensure we don't stitch to ourselves or go backwards
            if candidate and candidate.getID() != curr_lane.getID():
                next_internal = candidate

        # Stitch if found
        if next_internal:
            new_shape = next_internal.getShape()
            if full_coords[-1] == new_shape[0]:
                full_coords.extend(new_shape[1:])
            else:
                full_coords.extend(new_shape)
            curr_lane = next_internal
        else:
            break 
            
    return full_coords

def check_intersection(shape1, shape2):
    """
    Checks if two polylines (lists of (x,y) tuples) intersect.
    """
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def segment_intersect(p1, p2, p3, p4):
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    # Iterate through all segments of curve 1
    for i in range(len(shape1) - 1):
        p1, p2 = shape1[i], shape1[i+1]
        
        # Iterate through all segments of curve 2
        for j in range(len(shape2) - 1):
            p3, p4 = shape2[j], shape2[j+1]
            
            if segment_intersect(p1, p2, p3, p4):
                return True
    return False


def get_adjacency_matrices(net_file):
    """
    Parses a SUMO .net files into two adjacency matrices:
    - Flow Matrix: N x N matrix of connections that bleed into each other
    - Conflit Matrix: N x N matrix of connections that intersect, and that one party would have to give way. Location of intersection not provided for now

    Also returns nodes, which is a list of strings, which is the order in which the matrices are constructed
    """

    net = sumolib.net.readNet(net_file, withInternal=True) #Grab network
    tls = net.getTrafficLights()[0] #Grabs the traffic light

    all_lanes = set() #Create a fixed set of unique lanes

    #Probably not the most efficient way to iterate... but at this low level speed is not important for a one-off function
    for conn in tls.getConnections(): #For each connection in the traffic light
        in_lane_id = conn[0].getID() #Get the INPUT lane
        out_lane_id = conn[1].getID() #Get the OUTPUT lane
        
        all_lanes.add(in_lane_id) #Add each. Sets automatically filter duplicates
        all_lanes.add(out_lane_id)
        
        for lane_conn in conn[0].getOutgoing(): #look at the incoming lane's outgoing connection
            if lane_conn.getToLane().getID() == out_lane_id:
                via_id = lane_conn.getViaLaneID() #Find the internal lane
                if via_id:
                    all_lanes.add(via_id)
                break

    nodes = sorted(list(all_lanes)) #Ensure consistent ordering
    node_to_idx = {lane: i for i, lane in enumerate(nodes)} #Create a vector that represents indexes of each node

    print(f"Graph Construction: Found {len(nodes)} nodes (Lanes) out of 26.")

    adj_flow = np.eye(len(nodes)) #Create a 2D matrix for the flow representation

    for lane_id in nodes: #For each node/lane
        u = node_to_idx[lane_id] #Grab the corresponding Lane ID for this node
        lane_obj = net.getLane(lane_id)
        
        for conn in lane_obj.getOutgoing(): #For each of these outgoing connections
            target_id = conn.getToLane().getID() #Get the output node... Not applicable if this node is output node
            via_id = conn.getViaLaneID() #Get the internal node.. not applicable if this node is internal node or output node
            
            if via_id in node_to_idx: #This should always be true for the closed intersection we have
                v = node_to_idx[via_id] #Insert internal node into flow matrix
                adj_flow[u, v] = 1.0
                
            elif target_id in node_to_idx: #This will always be true as long as its not an input lane
                v = node_to_idx[target_id] #Insert Output
                adj_flow[u, v] = 1.0

    adj_conf = np.zeros((len(nodes), len(nodes))) #Dont use eye because conflict does not occur with itself
    internal_lanes = [l for l in nodes if l.startswith(":")] #Grab every internal lane... junction lanes always start with :

    lane_destinations = {} #For merging
    lane_sources = {} #For Diverging

    stitched_shapes = {}
    
    for l_id in internal_lanes:
        lane_obj = net.getLane(l_id)
        
        if lane_obj.getOutgoing():
            lane_destinations[l_id] = lane_obj.getOutgoing()[0].getToLane().getID()
        if lane_obj.getIncoming():
            lane_sources[l_id] = lane_obj.getIncoming()[0].getID()

        stitched_shapes[l_id] = get_full_internal_shape(lane_obj, net)

    for l1_id in internal_lanes:
        for l2_id in internal_lanes: #For every pair of internal lanes
            if l1_id == l2_id: continue #Don't care if they are the same lane
            
            u = node_to_idx[l1_id]
            v = node_to_idx[l2_id]

            shape1 = stitched_shapes[l1_id]
            shape2 = stitched_shapes[l2_id]

            dest1 = lane_destinations.get(l1_id) #Determine Destination for Merge
            dest2 = lane_destinations.get(l2_id)

            src1 = lane_sources.get(l1_id) #Determing source for diverge
            src2 = lane_sources.get(l2_id)

            if check_intersection(shape1, shape2): #If there is a proper intersection
                 adj_conf[u, v] = 1.0 #We assign a high conflict score
                 adj_conf[v, u] = 1.0

            elif (dest1 and dest2) and (dest1 == dest2): #If there is a merge
                 adj_conf[u, v] = 0.6 #We assign a slightly lower, but still present, score
                 adj_conf[v, u] = 0.6

            elif (src1 and src2) and (src1 == src2): #If there is a diverge
                adj_conf[u, v] = 0.2 #Assign a very low score signalling that it could be blocking
                adj_conf[v, u] = 0.2

    return nodes, torch.tensor(adj_flow, dtype=torch.float32), torch.tensor(adj_conf, dtype=torch.float32)

def plot_all_internal_lanes(net_path):
    print("Generating Visual Debugger...")
    net = sumolib.net.readNet(net_path, withInternal=True)

    # --- HELPER FUNCTION (Locally defined to ensure it has access to 'net') ---
    def get_stitched_shape_for_plot(lane_obj, start_map):
        raw_shape = lane_obj.getShape()
        full_coords = list(raw_shape)
        curr_lane = lane_obj
        
        for _ in range(10):
            next_internal = None
            
            # 1. Topology Check
            for conn in curr_lane.getOutgoing():
                target = conn.getToLane()
                if target.getID().startswith(":"):
                    next_internal = target
                    break
            
            # 2. Geometry Fallback (Crucial for the plot to look right!)
            if not next_internal:
                last_point = full_coords[-1]
                # Use strict rounding to match your working matrix logic
                key = (round(last_point[0], 2), round(last_point[1], 2))
                candidate = start_map.get(key)
                if candidate and candidate.getID() != curr_lane.getID():
                    next_internal = candidate

            if next_internal:
                new_shape = next_internal.getShape()
                if full_coords[-1] == new_shape[0]:
                    full_coords.extend(new_shape[1:])
                else:
                    full_coords.extend(new_shape)
                curr_lane = next_internal
            else:
                break
        return full_coords
    # -----------------------------------------------------------------------

    # Build the Geometry Map (Required for the fallback)
    start_map = {}
    internal_lanes = []
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            if lane.getID().startswith(":"):
                internal_lanes.append(lane)
                start_coord = lane.getShape()[0]
                key = (round(start_coord[0], 2), round(start_coord[1], 2))
                start_map[key] = lane

    # Identify Heads
    secondary_ids = set()
    for l in internal_lanes:
        for conn in l.getOutgoing():
            target_id = conn.getToLane().getID()
            if target_id.startswith(":"):
                secondary_ids.add(target_id)
                
    start_nodes = [l for l in internal_lanes if l.getID() not in secondary_ids]
    print(f"Plotting {len(start_nodes)} stitched lane chains...")

    plt.figure(figsize=(10, 10))
    for lane in start_nodes:
        # Use the robust local helper
        shape = get_stitched_shape_for_plot(lane, start_map)
        xs, ys = zip(*shape)
        color = (random.random(), random.random(), random.random())
        
        plt.plot(xs, ys, color=color, linewidth=2, label=lane.getID(), alpha=0.8)
        plt.text(xs[0], ys[0], lane.getID(), fontsize=8, color=color, fontweight='bold')
        plt.scatter(xs, ys, color=color, s=10)

    plt.title("Full Stitched Internal Lanes")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    NET_PATH = "environment/basic_intersection.net.xml"
    
    # Helper to print matrix cleanly
    def print_labeled_matrix(matrix, labels, title):
        print(f"\n--- {title} ---")
        print("      ", end="") 
        for i in range(len(labels)):
            print(f"{i:^5}", end="") # Widen column for floats
        print()
        
        rows, cols = matrix.shape
        for r in range(rows):
            label = f"{r}: {labels[r]}"
            print(f"{label:<15} [", end="")
            for c in range(cols):
                val = matrix[r, c].item()
                if val == 0:
                    print(f"  .  ", end="")
                else:
                    print(f" {val:.1f} ", end="")
            print("]")

    try:
        nodes, flow, conf = get_adjacency_matrices(NET_PATH)
        
        print("\n=== MATRIX DEBUGGER ===")
        print(f"Total Nodes: {len(nodes)}")
        
        # 1. Print List of Nodes (Legend)
        print("\n--- Node Legend ---")
        for i, node in enumerate(nodes):
            print(f"{i}: {node}")

        # 2. Print Flow Matrix (Connectivity)
        print_labeled_matrix(flow, nodes, "FLOW MATRIX (1 = Connection)")
        
        # 3. Print Conflict Matrix (Danger)
        print_labeled_matrix(conf, nodes, "CONFLICT MATRIX (1 = Crash/Merge)")
        
        rows, cols = conf.shape
        count = 0
        seen = set()
        for r in range(rows):
            for c in range(cols):
                if conf[r, c] > 0 and r != c:
                    pair = tuple(sorted((r, c)))
                    if pair not in seen:
                        seen.add(pair)
                        count += 1
        print(f"Total Unique Pairs: {count}")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    #plot_all_internal_lanes(NET_PATH) #Doesnt work at the moment
