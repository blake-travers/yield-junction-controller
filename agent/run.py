import os
import sys
import traci
import sumolib
import random
from agent.vehicle import Vehicle
from agent.generate_phases import generate_rule_based_phases 

# --- CONFIG ---
SUMO_CFG = "environment/sim.sumocfg"
NET_FILE = "environment/basic_intersection.net.xml"
SUMO_CMD = ["sumo-gui", "-c", SUMO_CFG]

# --- KEYBOARD INPUT ---
try:
    import msvcrt # Windows
    def is_change_triggered():
        if msvcrt.kbhit():
            return msvcrt.getch().lower() == b't'
        return False
except ImportError:
    import select # Linux/Mac
    def is_change_triggered():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower() == 't'
        return False

# --- YELLOW LIGHT HELPER ---
def get_yellow_state(current, target):
    y = list(current)
    for i in range(len(current)):
        # Green -> Red = Yellow
        if (current[i].lower() == 'g') and (target[i].lower() == 'r'):
            y[i] = 'y'
        # Red -> Green = Red (Safety Buffer)
        elif (current[i].lower() == 'r') and (target[i].lower() == 'g'):
            y[i] = 'r'
    return "".join(y)

# --- MAIN EXECUTION ---
print(f"Config: {SUMO_CFG}")

# 1. GET THE REAL PHASES
print("Generating Rule-Based Phases...")
VALID_PHASES = generate_rule_based_phases(NET_FILE)
print(f">> Successfully loaded {len(VALID_PHASES)} valid phases.")

# 2. START SIMULATION
traci.start(SUMO_CMD)
step = 0
tls_id = traci.trafficlight.getIDList()[0]

active_vehicles = {}
target_phase = None
is_transitioning = False
transition_timer = 0
YELLOW_DUR = 4

print(">> Simulation Started. Press 't' in terminal to switch phase.")

while step < 3600:
    traci.simulationStep()
    
    # --- PHASE SWITCHING LOGIC ---
    if is_transitioning:
        transition_timer -= 1
        if transition_timer <= 0:
            traci.trafficlight.setRedYellowGreenState(tls_id, target_phase)
            is_transitioning = False
            print(f"   [TLS] Green Active: {target_phase}")
            
    else:
        # Only switch if not currently transitioning
        if is_change_triggered():
            current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            
            # Pick a RANDOM phase from your verified list
            target_phase = random.choice(VALID_PHASES)
            
            # Don't pick the same one we are currently in
            while target_phase == current_state:
                target_phase = random.choice(VALID_PHASES)
            
            # Create Transition
            yellow_state = get_yellow_state(current_state, target_phase)
            traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
            
            is_transitioning = True
            transition_timer = YELLOW_DUR
            print(f"   [TLS] Transitioning...")

    # --- VEHICLE TRACKING ---
    # (Standard vehicle update logic)
    current_ids = set(traci.vehicle.getIDList())
    known_ids = set(active_vehicles.keys())
    departed = known_ids - current_ids
    for vid in departed: del active_vehicles[vid]
        
    for vid in current_ids:
        if vid in active_vehicles:
            active_vehicles[vid].update()
        else:
            new_car = Vehicle(vid)
            new_car.update()
            active_vehicles[vid] = new_car

    # Debug Print (Once every 100 steps)
    if step % 100 == 0:
        count = len(active_vehicles)
        print(f"\n[Step {step}] Active Vehicles: {count}")

    step += 1

traci.close()