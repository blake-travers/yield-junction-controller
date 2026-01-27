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

step = 0
active_vehicles = {} # { "id": VehicleObject }

while step < 3600:
    traci.simulationStep()
    current_ids = set(traci.vehicle.getIDList())
    
    # Remove departed
    known_ids = set(active_vehicles.keys())
    departed = known_ids - current_ids
    for vid in departed:
        del active_vehicles[vid]
        
    # Update or Spawn
    for vid in current_ids:
        if vid in active_vehicles:
            active_vehicles[vid].update()
        else:
            new_car = Vehicle(vid) 
            new_car.update()
            active_vehicles[vid] = new_car
            
    step += 1

traci.close()