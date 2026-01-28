import os
import sys
import traci
import sumolib
from agent.vehicle import Vehicle

SUMO_CFG = "environment/sim.sumocfg"
NET_FILE = "environment/basic_intersection.net.xml"
ROU_FILE = "environment/traffic.rou.xml"
SUMO_CMD = ["sumo-gui", "-c", SUMO_CFG] #Pass full config path

print(f"Config: {SUMO_CFG}")

traci.start(SUMO_CMD)
step = 0
active_vehicles = {} # { "id": VehicleObject }

while step < 3600:
    traci.simulationStep()
    current_ids = set(traci.vehicle.getIDList())
    
    #Remove departed vehicles
    known_ids = set(active_vehicles.keys())
    departed = known_ids - current_ids
    for vid in departed:
        del active_vehicles[vid]
        
    for vid in current_ids:
        if vid in active_vehicles:
            active_vehicles[vid].update() #Update existing vehicles
        else:
            new_car = Vehicle(vid) #If a new vehicle, add it to the list
            new_car.update()
            active_vehicles[vid] = new_car

        if step % 100 == 0:
                count = len(active_vehicles)
                print(f"\n[Step {step}] Active Vehicles: {count}")
                
                if count > 0:
                    # Updated Header to match new vector components
                    print(f"{'ID':<10} | {'Pos(N)':<6} | {'Spd(N)':<6} | {'Dir [L,S,R]':<6} | {'Lane ID'}")
                    print("-" * 70)
                    
                    for i, (vid, car) in enumerate(active_vehicles.items()):
                        if i >= 5: break 
                        
                        vec = car.toVector() 
                        
                        # Deconstruct vector for cleaner printing
                        # vec = [norm_pos, lane_idx, norm_speed, norm_wait, L, S, R]
                        n_pos  = round(vec[0], 2)
                        n_spd  = round(vec[1], 2)
                        dir_vec = vec[2:] # The last 3 are direction
                        lane_id = car.getLaneID()
                        
                        print(f"{vid:<10} | {n_pos:<6} | {n_spd:<6} | {str(dir_vec):<6} | {lane_id}")
    step += 1

traci.close()