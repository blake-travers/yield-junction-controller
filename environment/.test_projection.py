#NOTE: Entirely AI generated test case

import pygame
import numpy as np
from intersection_network import IntersectionGraph, Direction
from vehicle import Vehicle, VehicleAction

# --- CONFIG ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 10
FPS = 60

# --- COLORS ---
COLOR_BACKGROUND = (30, 30, 30)
COLOR_ROAD = (80, 80, 80)
COLOR_REAL_CAR = (0, 255, 0)   # Green
# Gradient for ghosts: 1s (Dark Blue) -> 4s (Light Blue/Cyan)
COLOR_GHOSTS = [
    (0, 0, 150),    # 1.0s - Dark Blue
    (0, 0, 255),    # 2.0s - Blue
    (0, 150, 255),  # 3.0s - Light Blue
    (0, 255, 255)   # 4.0s - Cyan
]

def world_to_screen(x, y):
    screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
    screen_y = int(SCREEN_HEIGHT / 2 - y * SCALE)
    return screen_x, screen_y

def run_test():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)

    # 1. Setup Graph
    graph = IntersectionGraph()
    graph.build_lanes()

    # 2. Setup Vehicle (Turning Left to test curves)
    #    Spawn on North Queue -> Left -> East Exit
    v_test = Vehicle("inqN1", Direction.LEFT, graph, start_speed=0.0)
    
    #    Place car 5 meters from the end of the first lane segment
    #    This ensures it MUST cross into the next segment (the curve)
    u, v = v_test.route[0], v_test.route[1]
    edge_len = graph.G.edges[u, v]['geometry'].length
    v_test.position = edge_len - 5.0 

    timesteps = [1.0, 2.0, 3.0, 4.0]

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(COLOR_BACKGROUND)

        # --- DRAW MAP ---
        for u, v, data in graph.G.edges(data=True):
            if 'geometry' in data:
                pts = [world_to_screen(x, y) for x, y in data['geometry'].coords]
                if len(pts) > 1:
                    pygame.draw.lines(screen, COLOR_ROAD, False, pts, 2)
                    # Draw node connections as small dots
                    pygame.draw.circle(screen, (100, 50, 50), pts[-1], 3)

        # --- DRAW REAL CAR (Green) ---
        real_box = v_test.get_bounding_box()
        real_pts = [world_to_screen(x, y) for x, y in real_box.exterior.coords]
        pygame.draw.polygon(screen, COLOR_REAL_CAR, real_pts)
        
        # Label Real Car
        cx, cy = world_to_screen(*v_test.get_global_position()[:2])
        label = font.render("START", True, COLOR_REAL_CAR)
        screen.blit(label, (cx + 20, cy))

        # --- DRAW GHOSTS (Projected) ---
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

        for i, t in enumerate(timesteps):
            # Calculate Projection
            ghost_poly = v_test._get_projected_location(VehicleAction.ACCEL, time=t)
            
            if ghost_poly:
                pts = [world_to_screen(x, y) for x, y in ghost_poly.exterior.coords]
                
                # Pick color from list (with some transparency)
                base_color = COLOR_GHOSTS[i]
                final_color = (base_color[0], base_color[1], base_color[2], 150) # Alpha=150
                
                pygame.draw.polygon(s, final_color, pts)
                
                # Draw Outline
                pygame.draw.lines(s, base_color, True, pts, 2)

                # Draw Label (e.g., "1.0s")
                # Find top-left corner of the box for label placement
                label_x = min(p[0] for p in pts)
                label_y = min(p[1] for p in pts)
                
                txt = font.render(f"{t}s", True, (255, 255, 255))
                # Draw text on main screen (not alpha surface) for clarity
                screen.blit(txt, (label_x, label_y - 20))

        screen.blit(s, (0,0))

        # --- LEGEND ---
        legend_y = 10
        pygame.draw.rect(screen, (50, 50, 50), (10, 5, 200, 120))
        
        info1 = font.render(f"Speed: {v_test.speed} m/s", True, (255, 255, 255))
        screen.blit(info1, (20, legend_y))
        legend_y += 25

        info2 = font.render("Green: Real Car (t=0)", True, COLOR_REAL_CAR)
        screen.blit(info2, (20, legend_y))
        legend_y += 25
        
        for i, t in enumerate(timesteps):
            txt = font.render(f"Blue/Cyan: t={t}s", True, COLOR_GHOSTS[i])
            screen.blit(txt, (20, legend_y))
            legend_y += 20

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run_test()