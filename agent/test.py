import sumolib
import matplotlib.pyplot as plt
import random

def get_full_internal_shape(lane_obj):
    """
    Iteratively follows internal lane connections to build the 
    FULL shape of the movement, stitching segments together.
    """
    raw_shape = lane_obj.getShape()
    full_coords = list(raw_shape)
    
    curr_lane = lane_obj
    
    # Safety loop to prevent infinite cycling (max 10 segments per turn)
    for _ in range(10):
        outgoing = curr_lane.getOutgoing()
        if not outgoing:
            break
            
        # Find the next segment in the chain (must be internal)
        next_internal = None
        for conn in outgoing:
            target = conn.getToLane()
            if target.getID().startswith(":"):
                next_internal = target
                break
        
        if next_internal:
            new_shape = next_internal.getShape()
            
            # Stitch coordinates: avoid duplicating the join point
            if full_coords[-1] == new_shape[0]:
                full_coords.extend(new_shape[1:])
            else:
                full_coords.extend(new_shape)
            
            curr_lane = next_internal
        else:
            break # End of the internal chain
            
    return full_coords

def plot_all_internal_lanes(net_path):
    # 1. Load the Network with Internal edges
    net = sumolib.net.readNet(net_path, withInternal=True)
    
    # 2. Collect all internal lanes by iterating Edges first
    internal_lanes = []
    # getEdges(withInternal=True) is required to see the :J edges
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            if lane.getID().startswith(":"):
                internal_lanes.append(lane)

    # 3. Identify "Head" lanes
    # We only want to plot the start of a chain (e.g. :J_3_0), not the middle (:J_3_1)
    # Logic: If a lane is the target of another internal lane, it is NOT a head.
    secondary_ids = set()
    for l in internal_lanes:
        for conn in l.getOutgoing():
            target_id = conn.getToLane().getID()
            if target_id.startswith(":"):
                secondary_ids.add(target_id)
                
    # The heads are the ones that are NOT in the secondary set
    start_nodes = [l for l in internal_lanes if l.getID() not in secondary_ids]

    print(f"Found {len(start_nodes)} distinct internal movement chains.")

    # 4. Plot
    plt.figure(figsize=(10, 10))
    
    for lane in start_nodes:
        lane_id = lane.getID()
        
        # Get the stitched shape
        shape = get_full_internal_shape(lane)
        xs, ys = zip(*shape)
        
        # Random color for distinction
        color = (random.random(), random.random(), random.random())
        
        plt.plot(xs, ys, color=color, linewidth=2, label=lane_id, alpha=0.8)
        
        # Label the start of the line
        plt.text(xs[0], ys[0], lane_id, fontsize=8, color=color, fontweight='bold')
        
        # Show points to see where segments were stitched
        plt.scatter(xs, ys, color=color, s=10)

    plt.title("Full Stitched Internal Lanes (Mathematical Centerlines)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis('equal') # Crucial for seeing the true intersection geometry
    plt.show()

if __name__ == "__main__":
    NET_PATH = "environment/basic_intersection.net.xml"
    plot_all_internal_lanes(NET_PATH)