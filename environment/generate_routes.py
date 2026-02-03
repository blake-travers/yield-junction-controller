import os
import sys
import sumolib
import random
import subprocess

def generate_routes(seed, length, frequency):
    net_file = "environment/basic_intersection.net.xml"
    parsed_routes = "environment/parsed.rou.xml"
    prelim_routes = "environment/prelim.rou.xml"

    try: #Setup Path
        sumo_binary = sumolib.checkBinary('sumo')
        sumo_home = os.path.dirname(os.path.dirname(sumo_binary))
        random_trips_path = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    except Exception:
        if 'SUMO_HOME' in os.environ:
             random_trips_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
        else:
             sys.exit("Error: Could not find SUMO_HOME.")

    attrs = "departLane='best' type='DEFAULT_TYPE'" #No Lane Changing, Overwrite Type Later
    
    cmd = [ #Regular Route gen command
        sys.executable, random_trips_path,
        "-n", net_file,
        "-r", parsed_routes,
        "-o", prelim_routes,
        "-e", str(length),
        "-p", str(frequency),
        "--seed", str(seed),
        "--trip-attributes", attrs
    ]

    try: #Generate the Route
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("Error: Route generation failed.", e.stderr)
        sys.exit(1)
    
    vtypes = [  #Define Car Presets
        '    <vType accel="2.6" decel="4.5" maxSpeed="15" sigma="0.2" length="5.0" width="2.0" id="DEFAULT" speedFactor="1" tau="1" color="1,1,0"/>', #Default Car (YELLOW)
        '    <vType accel="4.5" decel="6.0" maxSpeed="25" sigma="0.6" length="4.0" width="1.75" id="AGGRESIVE" speedFactor="normc(1.2,0.1)" tau="0.8" color="1,0,0"/>', #Agressive car (RED)
        '    <vType accel="2.0" decel="4.5" maxSpeed="12" sigma="1.0" length="5.5" width="2.25" id="SLUGGISH" speedFactor="normc(0.95,0.1)" tau="2.0" color="0,0,1"/>', #Sluggish car (BLUE)
        '    <vType accel="1.2" decel="2.5" maxSpeed="15" sigma="0.3" length="10"  width="2.75" id="LARGE" speedFactor="0.8" tau="1.5" color="0,1,0"/>' #Slow car (GREEN)
    ]
    
    type_ids = ["DEFAULT", "AGGRESIVE", "SLUGGISH", "LARGE"]
    weights = [0.4, 0.2, 0.2, 0.2] #40% Default, 20% Aggressive, 20% Sluggish, 20% Slow

    try:
        with open(parsed_routes, 'r') as f: #Open route file
            lines = f.readlines()
        
        new_lines = []
        routes_tag_found = False

        for line in lines:
            if "<routes" in line and not routes_tag_found: #Inject Definitions
                new_lines.append(line)
                new_lines.append("\n".join(vtypes) + "\n")
                routes_tag_found = True
            
            elif "DEFAULT_TYPE" in line: #For each Vehicle
                chosen_type = random.choices(type_ids, weights=weights, k=1)[0] #Choose a random vehicle type
                new_line = line.replace("DEFAULT_TYPE", chosen_type)
                new_lines.append(new_line) #Replace the type with this one
             
            else:
                new_lines.append(line)
        
        with open(parsed_routes, 'w') as f: #Write back to file
            f.writelines(new_lines)
            
    except IOError as e:
        print(f"Error processing route file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_routes(random.randint(0, 1000000), 360, 4)