#NOTE: Entirely AI generated test case

import pygame
import numpy as np  
from shapely.geometry import Polygon, MultiPolygon
from intersection_network import IntersectionGraph, Direction
from vehicle import Vehicle, VehicleAction

# --- CONFIG ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCALE = 10
FPS = 60

# --- COLORS ---
COLOR_BG = (30, 30, 30)
COLOR_ROAD = (80, 80, 80)
COLOR_REAL_CAR = (0, 255, 0)     # Green
COLOR_BRAKE_LIMIT = (255, 0, 0)  # Red (Min Dist)
COLOR_ACCEL_LIMIT = (0, 0, 255)  # Blue (Max Dist)
COLOR_SNAKE = (255, 165, 0)      # Orange (The Union)

def world_to_screen(x, y):
    screen_x = int(SCREEN_WIDTH / 2 + x * SCALE)
    screen_y = int(SCREEN_HEIGHT / 2 - y * SCALE)
    return screen_x, screen_y

def draw_shapely_poly(surface, poly, color, alpha=100, outline=True):
    """Helper to draw Shapely polygons (handles simple and multipolygons)"""
    if poly.is_empty: return

    # Handle MultiPolygons (though unlikely here)
    if isinstance(poly, MultiPolygon):
        geoms = poly.geoms
    else:
        geoms = [poly]

    for geom in geoms:
        if not hasattr(geom, 'exterior'): continue
        
        pts = [world_to_screen(x, y) for x, y in geom.exterior.coords]
        
        # Draw Fill (with Alpha)
        s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        # Add alpha to color tuple if not present
        if len(color) == 3:
            fill_color = (*color, alpha)
        else:
            fill_color = color
            
        pygame.draw.polygon(s, fill_color, pts)
        surface.blit(s, (0,0))
        
        # Draw Outline
        if outline:
            pygame.draw.lines(surface, color, True, pts, 2)

def run_test():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)

    # 1. Setup
    graph = IntersectionGraph()
    graph.build_lanes()

    # Spawn Car Turning Left (Best for testing geometry)
    v_test = Vehicle("inqN1", Direction.LEFT, graph, start_speed=12.0)
    
    # Place car 10m before the intersection to see the curve traversal
    u, v = v_test.route[0], v_test.route[1]
    edge_len = graph.G.edges[u, v]['geometry'].length
    v_test.position = edge_len - 10.0 

    # We test for a 2.0 second horizon
    DT = 2.0

    print(f"Test Config: Speed={v_test.speed}m/s, dt={DT}s")

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Optional: Allow user to change speed with Up/Down keys
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: v_test.speed += 1.0
                if event.key == pygame.K_DOWN: v_test.speed -= 1.0
                v_test.speed = max(0, v_test.speed)

        screen.fill(COLOR_BG)

        # --- DRAW MAP ---
        for u, v, data in graph.G.edges(data=True):
            if 'geometry' in data:
                pts = [world_to_screen(x, y) for x, y in data['geometry'].coords]
                if len(pts) > 1:
                    pygame.draw.lines(screen, COLOR_ROAD, False, pts, 2)
                    pygame.draw.circle(screen, (100, 50, 50), pts[-1], 3)

        # --- CALCULATE LOGIC ---
        
        # 1. Get the Union Snake (The Method we are testing)
        snake_union = v_test.get_possible_locations(DT)
        
        # 2. Get the Limits (For verification visual only)
        #    We manually call the private methods to see where the 'Tips' are
        dist_brake = v_test._calculate_projected_distance(VehicleAction.BRAKE, DT)
        dist_accel = v_test._calculate_projected_distance(VehicleAction.ACCEL, DT)
        
        box_brake = v_test._get_box_at_distance(dist_brake)
        box_accel = v_test._get_box_at_distance(dist_accel)

        # --- DRAWING ---

        # 1. Draw Snake (Bottom Layer)
        if snake_union:
            draw_shapely_poly(screen, snake_union, COLOR_SNAKE, alpha=80)

        # 2. Draw Real Car
        draw_shapely_poly(screen, v_test.get_bounding_box(), COLOR_REAL_CAR, alpha=255)

        # 3. Draw Limits (Wireframes on top)
        if box_brake:
            draw_shapely_poly(screen, box_brake, COLOR_BRAKE_LIMIT, alpha=0, outline=True)
        if box_accel:
            draw_shapely_poly(screen, box_accel, COLOR_ACCEL_LIMIT, alpha=0, outline=True)

        # --- TEXT INFO ---
        info_lines = [
            f"Speed: {v_test.speed:.1f} m/s",
            f"DT: {DT} s",
            f"Brake Dist: +{dist_brake:.1f} m (RED)",
            f"Accel Dist: +{dist_accel:.1f} m (BLUE)",
            "",
            "Controls: UP/DOWN to change speed"
        ]
        
        y_offset = 10
        for line in info_lines:
            txt = font.render(line, True, (255, 255, 255))
            screen.blit(txt, (10, y_offset))
            y_offset += 20

        # Draw Legend
        pygame.draw.rect(screen, COLOR_SNAKE, (10, y_offset + 10, 20, 20))
        screen.blit(font.render("Uncertainty Union", True, (200, 200, 200)), (40, y_offset + 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run_test()