import numpy as np
import uuid
from enum import Enum
from intersection_network import Direction
from shapely.geometry import Point

class VehicleAction(Enum):
    BRAKE = 0
    COAST = 1
    ACCEL = 2

class Vehicle:
    def __init__(self, spawn_node, direction:Direction, graph, length = 5, width = 3, max_speed = 14, accel = 3, brake = 5, start_speed = 0):
        self.id = str(uuid.uuid4())[:8]

        self.length = length
        self.width = width
        self.max_speed = max_speed
        self.accel = accel
        self.brake = brake

        self.position = 0 #Distance travelled along the current edge of the route
        self.speed = start_speed
        self.done = False

        self.graph = graph #Represents the state of the graph... used to determine route etc
        self.route = self._calculate_route(spawn_node, direction)
        self.route_index = 0
        self._update_current_nodes()


    def _calculate_route(self, spawn_node, direction):
        """
        Builds the list of nodes based upon the starting node of the car and the direction it wishes to go

        Highly dependant on node naming convention
        """
        
        intersection_entry = spawn_node.replace("q", "") #identify the name of the second node
        if intersection_entry is None:
            raise ValueError(f"No path found for {spawn_node} into {intersection_entry}")

        intersection_exit = None
        for _, end_node, data in self.graph.G.out_edges(intersection_entry, data=True): #Get all edges for the intersection entry node
            if data.get("turn_type") == direction.value: #If that edge corresponds with the direction we want to go
                intersection_exit = end_node
                break

        if intersection_exit is None:
            raise ValueError(f"No path found for {intersection_entry} turning {direction.value}")
        
        despawn_node = intersection_exit.replace("out", "outq")

        if despawn_node is None:
            raise ValueError(f"No path found for {intersection_exit} into {despawn_node}")
        
        return [spawn_node, intersection_entry, intersection_exit, despawn_node]
        
    def apply_action(self, action: VehicleAction, dt):
        """
        Applies Acceleration / Braking physics to the Vehicle
        """

        if action == VehicleAction.ACCEL:
            accel = self.accel
        elif action == VehicleAction.BRAKE:
            accel = -self.brake
        else:
            accel = 0
        
        self.speed += accel * dt #Update the velocity. Note that dt is the control meaning it is realistic
        self.speed = np.clip(self.speed, 0, self.max_speed) #Clamp to ensure reasonable values

        dist = self.speed * dt #Update position based upon velocity
        self.position += dist
        
        self._check_edge_transition()

    def _check_edge_transition(self):
        """
        Checks to see if the Vehicle has exceeded the current edge and needs to go to the next one
        """
        u, v = self.current_start_node, self.current_end_node
        edge_len = self.graph.G.edges[u, v]['geometry'].length
        
        if self.position >= edge_len:
            excess = self.position - edge_len
            
            if self.route_index < len(self.route) - 2: #If there is a next edge
                self.route_index += 1 #Update the current index
                self._update_current_nodes() #Update nodes
                self.position = excess #Transfer excess position over to this next node
            else: #Else we are done
                self.done = True
    
    def _update_current_nodes(self):
        self.current_start_node = self.route[self.route_index] #Start and end node dictate the current edge
        self.current_end_node = self.route[self.route_index+1]

    def get_global_position(self):
        """
        Gets the global coordinates of the vehicle, for pygame rendering
        """
        data = self.graph.G.get_edge_data(self.current_start_node, self.current_end_node) #Get the edge
        
        line = data['geometry']
        # Clamp to avoid rounding errors at end of line
        safe_pos = min(self.position, line.length - 0.001)
        point = line.interpolate(safe_pos)
        
        # Calculate Heading (Angle)
        next_point = line.interpolate(safe_pos + 1.0)
        dx = next_point.x - point.x
        dy = next_point.y - point.y
        angle = np.arctan2(dy, dx) # Radians
        
        return point.x, point.y, angle