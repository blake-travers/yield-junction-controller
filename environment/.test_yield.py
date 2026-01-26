import pygame
import numpy as np
import random
from intersection_network import IntersectionGraph, Direction
from vehicle import Vehicle, VehicleAction

# --- CONFIG ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 10
FPS = 60

# --- COLORS ---
COLOR_ROAD = (80, 80, 80)
COLOR_BG = (30, 30, 30)
COLOR_SAFE = (0, 255, 0)     # Green = Accelerating
COLOR_BRAKING = (255, 0, 0)  # Red = Braking

def world_to_screen(x, y):
    screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
    screen_y = int(SCREEN_HEIGHT / 2 - y * SCALE)
    return screen_x, screen_y

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    graph = IntersectionGraph()
    graph.build_lanes()
 
    vehicles = []

    # PRE-SCENARIO: Spawn two cars on a collision course
    # Car 1: From North going Straight
    v1 = Vehicle("inqN1", Direction.STRAIGHT, graph, start_speed=10.0)
    # Car 2: From East going Straight (Will cross v1's path)
    v4 = Vehicle("inqS2", Direction.RIGHT, graph, start_speed=10.0)
    
    vehicles.append(v1)
    vehicles.append(v4)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # SPACE BAR: Spawn random traffic to test chaos
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                try:
                    r_node = random.choice(["inqN1", "inqN2", "inqS1", "inqS2", "inqW", "inqE"])
                    r_dir = random.choice(list(Direction))
                    v_new = Vehicle(r_node, r_dir, graph, start_speed=8 + random.random()*4)
                    vehicles.append(v_new)
                except ValueError:
                    pass

        # --- LOGIC UPDATE ---
        for v in vehicles[:]:
            if v.done:
                vehicles.remove(v)
            else:
                # 1. THE BRAIN: Decide Action
                # This calls your new conservative logic
                action = v.action_decision(vehicles)
                
                # 2. THE BODY: Apply Action
                v.apply_action(action, dt)
                
                # Store action for visualization color
                v.last_action = action 

        # --- DRAWING ---
        screen.fill(COLOR_BG)

        # 1. Draw Map
        for u, v_node, data in graph.G.edges(data=True):
            if 'geometry' in data:
                pts = [world_to_screen(x, y) for x, y in data['geometry'].coords]
                if len(pts) > 1:
                    pygame.draw.lines(screen, COLOR_ROAD, False, pts, 2)

        # 2. Draw Vehicles
        for v in vehicles:
            box = v.get_bounding_box()
            pts = [world_to_screen(x, y) for x, y in box.exterior.coords]
            
            # Color based on decision
            color = COLOR_SAFE
            if hasattr(v, 'last_action') and v.last_action == VehicleAction.BRAKE:
                color = COLOR_BRAKING
            
            pygame.draw.polygon(screen, color, pts)
            
            # Optional: Draw ID or Speed
            # lbl = font.render(f"{v.speed:.1f}", True, (255,255,255))
            # screen.blit(lbl, pts[0])

        # Info
        count_txt = font.render(f"Vehicles: {len(vehicles)} (Green=Accelerate, Red=Brake)", True, (255, 255, 255))
        screen.blit(count_txt, (10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run_simulation()