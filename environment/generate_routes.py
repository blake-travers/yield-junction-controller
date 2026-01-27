import os
import sys
import sumolib

def generate_routes():

    net_file = "environment/basic_intersection.net.xml"
    route_file = "environment/traffic.rou.xml"

    try:
        # sumolib.checkBinary('sumo') gives path to bin/sumo
        # .rsplit(..., 2)[0] goes up two levels to find the SUMO_HOME root
        sumo_binary = sumolib.checkBinary('sumo')
        sumo_home = os.path.dirname(os.path.dirname(sumo_binary))
        random_trips_path = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    except Exception:
        # Fallback if standard install structure is different
        if 'SUMO_HOME' in os.environ:
             random_trips_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
        else:
             sys.exit("Error: Could not find SUMO_HOME. Please set it or ensure sumolib is installed.")

    # 4. Run the command using absolute paths
    # We wrap paths in quotes "" to handle any potential spaces in folder names
    cmd = f'python "{random_trips_path}" -n "{net_file}" -r "{route_file}" -e 3600 -p 2'

    print(f"Generating routes in: {base_path}")
    os.system(cmd)
    
    # 5. Verify it worked
    if os.path.exists(route_file):
        print("Success! 'traffic.rou.xml' is safely inside the environment folder.")
    else:
        print("Error: The file was not created. Check the command output above.")

if __name__ == "__main__":
    generate_routes()