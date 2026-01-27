import os
import sys
import sumolib

def print_full_graph_nodes():
    net_file = "environment/basic_intersection.net.xml"

    if not os.path.exists(net_file):
        sys.exit(f"Error: Could not find '{net_file}'")

    print(f"Reading: {net_file}")
    net = sumolib.net.readNet(net_file)
    
    tls_list = net.getTrafficLights()
    if not tls_list:
        print("Error: No Traffic Light found!")
        return

    tls = tls_list[0]
    print(f"\nJunction ID: {tls.getID()}")

    # 2. Get Controlled Links
    # Returns list of [inLane, outLane, linkIndex]
    connections = tls.getConnections()
    connections.sort(key=lambda x: x[2]) 

    # We will store the full flow structure here
    # Structure: { LinkIndex: { 'in': ID, 'internal': ID, 'out': ID, 'dir': '?' } }
    graph_structure = {}

    print("\n--- COMPLETE GNN NODE STRUCTURE ---")
    print(f"{'Link':<5} | {'Incoming (Queue)':<18} -> {'Internal (Crossing)':<20} -> {'Outgoing (Exit)':<18} | {'Dir'}")
    print("-" * 100)

    unique_internal_nodes = set()
    unique_outgoing_nodes = set()

    for conn in connections:
        link_idx = conn[2]
        in_lane_obj = conn[0]
        out_lane_obj = conn[1]
        
        in_id = in_lane_obj.getID()
        out_id = out_lane_obj.getID()
        
        # 3. Find the Internal Lane (The "Via" Lane)
        # We look at the Incoming Lane's physical connections to find the specific path
        via_id = "None"
        dir_code = "?"
        
        for lane_conn in in_lane_obj.getOutgoing():
            if lane_conn.getToLane().getID() == out_id:
                dir_code = lane_conn.getDirection() # 's', 'r', 'l'
                via_id = lane_conn.getViaLaneID()   # This is the Internal Lane ID!
                break
        
        # Formatting for cleaner printing
        if not via_id: via_id = "Direct (No Internal)"
        
        print(f"{link_idx:<5} | {in_id:<18} -> {via_id:<20} -> {out_id:<18} | {dir_code}")

        # Collect unique IDs for your reference
        if via_id and "Direct" not in via_id:
            unique_internal_nodes.add(via_id)
        unique_outgoing_nodes.add(out_id)

    print("-" * 100)
    print(f"\nStats for GNN Dimensioning:")
    print(f"1. Incoming Nodes (Queues): {len(set(c[0].getID() for c in connections))}")
    print(f"2. Internal Nodes (Crossing): {len(unique_internal_nodes)} (IDs like :Cluster_...)")
    print(f"3. Outgoing Nodes (Exits):    {len(unique_outgoing_nodes)}")

    print(f"\nAll Internal Nodes (Add these to your GNN!):")
    print(sorted(list(unique_internal_nodes)))

if __name__ == "__main__":
    print_full_graph_nodes()