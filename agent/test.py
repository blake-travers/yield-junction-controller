import sumolib

NET_FILE = "environment/basic_intersection.net.xml"

def identify_link_zero():
    print(f"Loading {NET_FILE}...")
    net = sumolib.net.readNet(NET_FILE)
    
    # Get the Traffic Light Logic
    tls = net.getTrafficLights()[0] # Assuming you only have one junction 'J2'
    print(f"Inspecting TLS ID: {tls.getID()}")

    # Loop through all connections to find Index 0
    found = False
    for connection in tls.getConnections():
        # Connection format: [fromLane, toLane, linkIndex]
        in_lane = connection[0]
        out_lane = connection[1]
        link_index = connection[2]

        if link_index == 0:
            print("\n--- FOUND LINK INDEX 0 ---")
            print(f"From Lane:   {in_lane.getID()}")
            print(f"To Lane:     {out_lane.getID()}")
            print(f"Direction:   {connection[3] if len(connection) > 3 else 'Unknown'}")
            print("--------------------------\n")
            found = True
            break
            
    if not found:
        print("Could not find Link Index 0. That's weird.")

if __name__ == "__main__":
    identify_link_zero()