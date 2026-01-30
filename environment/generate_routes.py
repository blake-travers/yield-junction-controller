import os
import sys
import sumolib
import random 

def generate_routes(seed):
    net_file = "environment/basic_intersection.net.xml"
    route_file = "environment/traffic.rou.xml"

    try:
        sumo_binary = sumolib.checkBinary('sumo')
        sumo_home = os.path.dirname(os.path.dirname(sumo_binary))
        random_trips_path = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    except Exception:
        if 'SUMO_HOME' in os.environ:
             random_trips_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
        else:
             sys.exit("Error: Could not find SUMO_HOME.")

    # 2. Add the --seed flag and --quiet to keep logs clean
    # We use -p 2 (period) to define traffic density. 
    # Lower -p = higher density.
    cmd = (f'python "{random_trips_path}" '
           f'-n "{net_file}" '
           f'-r "{route_file}" '
           f'-e 360 '
           f'-p 2 '
           f'--seed {seed} '
           f'--trip-attributes "departLane=\'best\'" ')

    os.system(cmd)

if __name__ == "__main__":
    seed = random.randint(0, 1000000)
    generate_routes(seed)