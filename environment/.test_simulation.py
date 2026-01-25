#NOTE: Entirely AI generated test case

import pygame
import numpy as np
import random
from intersection_network import IntersectionGraph, Direction
from vehicle import Vehicle, VehicleAction

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 10
FPS = 60

# --- HELPER: Coordinate Conversion ---
def world_to_screen(x, y): #Converts coordinates
    screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
    screen_y = int(SCREEN_HEIGHT / 2 - y * SCALE)
    return screen_x, screen_y

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    graph = IntersectionGraph()
    graph.build_lanes()
 
    vehicles = []

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        #Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                spawned = False
                attempts = 0
                while not spawned and attempts < 10: #Try 10 times to find a valid combo, this should work first time ~80% of the time
                    try:
                        r_node = random.choice(["inqN1", "inqN2", "inqS1", "inqS2", "inqW", "inqE"])
                        r_dir = random.choice(list(Direction))
                        
                        v_new = Vehicle(r_node, r_dir, graph, start_speed=6 + random.random()*4) #This will automatically error if invalid route. Start speed is variable
                        vehicles.append(v_new)
                        spawned = True
                        print(f"Spawned: {r_node} going {r_dir.name}")
                        
                    except ValueError:
                        attempts += 1

        for v in vehicles[:]:
            if v.done:
                vehicles.remove(v)
            else:
                # For now, just accelerate blindly
                v.apply_action(VehicleAction.ACCEL, dt)

        # --- Drawing ---
        screen.fill((30, 30, 30)) # Dark background

        # 1. Draw Map (Static Edges)
        for u, v, data in graph.G.edges(data=True):
            if 'geometry' in data:
                # Convert shapely LineString to list of screen points
                screen_points = []
                for x, y in data['geometry'].coords:
                    screen_points.append(world_to_screen(x, y))
                
                if len(screen_points) > 1:
                    pygame.draw.lines(screen, (100, 100, 100), False, screen_points, 2)

        # 2. Draw Vehicles (Dynamic)
        for v in vehicles:
            box = v.get_bounding_box()
            
            # Convert world coordinates (meters) to screen coordinates (pixels)
            screen_points = []
            for x, y in box.exterior.coords:
                screen_points.append(world_to_screen(x, y))
            
            # Draw the filled car body (Green)
            pygame.draw.polygon(screen, (0, 255, 0), screen_points)
        

        pygame.display.flip()
        pygame.display.set_caption(f"Traffic Sim | Cars: {len(vehicles)}")

    pygame.quit()

if __name__ == "__main__":
    run_simulation()