import numpy as np
import uuid
from enum import Enum
from intersection_network import Direction
from shapely.geometry import Polygon
from shapely.ops import unary_union


class VehicleAction(Enum):
    BRAKE = 0
    ACCEL = 1

class Vehicle:
    def __init__(self, spawn_node, direction:Direction, graph, length = 5.5, width = 2.5, max_speed = 15, accel = 6, brake = 10, start_speed = 0):
        self.id = str(uuid.uuid4())[:8]

        self.length = length
        self.width = width
        self.max_speed = max_speed
        self.accel = accel
        self.brake = brake

        self.position = 0 #Distance travelled along the current edge of the route
        self.speed = start_speed
        self.done = False
        self.direction = direction

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
        else: #Should never happen apart from the very start (edge case)
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
        Gets the global coordinates & angle of the vehicle, for pygame rendering
        """
        data = self.graph.G.get_edge_data(self.current_start_node, self.current_end_node) #Get the edge
        
        line = data['geometry']
        # Clamp to avoid rounding errors at end of line
        safe_pos = min(self.position, line.length - 0.001)
        point = line.interpolate(safe_pos) #Pretty much 99% accurate interpolation
        
        # Calculate Heading (Angle)
        next_point = line.interpolate(safe_pos + 1.0)
        dx = next_point.x - point.x
        dy = next_point.y - point.y
        angle = np.arctan2(dy, dx) # Radians
        
        return point.x, point.y, angle
    
    def get_bounding_box(self, x = None, y = None, angle = None):
        """
        Calculates the 4 corners of the vehicle as a rotated Polygon.

        Bounding box is used for two purposes: 
        1. Simulation representation - shows the car rotation which while not neccesary is helpful
        2. Collision detection - bounding box is used to project its location if accelerating and make action decision to brake or accelerate from that
        """
        if x is None or y is None or angle is None:
            x, y, angle = self.get_global_position() #Get current coordinates and angle

        half_length = self.length / 2
        half_width = self.width / 2
        
        #    [Rear-Left, Rear-Right, Front-Right, Front-Left]
        corners_local = [
            (-half_length,  half_width), 
            (-half_length, -half_width), 
            ( half_length, -half_width), 
            ( half_length,  half_width)
        ]
        
        c, s = np.cos(angle), np.sin(angle) #Corners need to be rotated based upon the current vehicle angle
        
        corners_global = []
        for lx, ly in corners_local: #Use Sine and Cosine to determine locations.. basic trig
            rx = lx * c - ly * s + x
            ry = lx * s + ly * c + y
            corners_global.append((rx, ry))
            
        return Polygon(corners_global)


    def _calculate_projected_distance(self, action, time):
        """
        Helper function representing the projected distance given an time value and assuming a constant action throughout this time
        
        """
        if action == VehicleAction.ACCEL:
            accel = self.accel
            target_speed = self.max_speed
        else: #VehicleAction.BRAKE
            accel = -self.brake
            target_speed = 0.0

        time_to_limit = (target_speed - self.speed) / accel #Calculate the time till either speed reaches max or speed reaches 0 depending on action

        if time < time_to_limit: # Standard Kinematics - if projection constantly accelerates
            distance_travelled = (self.speed * time) + (0.5 * accel * time**2)
        else: #We either hit max speed or stop entirely sometime through the projection
            dist_to_limit = (target_speed**2 - self.speed**2) / (2 * accel) #Distance travelled below the limit
            time_at_limit = time - time_to_limit #Time travelled at the limit
            dist_at_limit = target_speed * time_at_limit #Distance travelled at the limit. At brake this is always zero because target speed is 0
            distance_travelled = dist_to_limit + dist_at_limit #Total distance travelled

        return distance_travelled
    
    def _get_box_at_distance(self, distance_travelled):
        """
        Helper function representing the bounding box at a specific distance from this car
        """

        remaining_dist = self.position + distance_travelled #This represents the total distance from the start of the node.

        current_edge_idx = self.route_index
        u = self.route[current_edge_idx]
        v = self.route[current_edge_idx + 1]
        line = self.graph.G.edges[u, v]['geometry']
        
        while remaining_dist > line.length: #If the distance is larger than the entire edge length. This while loop will run a maximum of 3 times (beacuse max route length of 4)
            if current_edge_idx + 2 >= len(self.route): #Ensure there is another edge, if not then:
                remaining_dist = line.length - 0.001
                break
            
            remaining_dist -= line.length #Remove the total length of the line from the remaining distance value, this will most of the time result in less than zero (but we clip)
            current_edge_idx += 1
            
            u = self.route[current_edge_idx] #Update to the new nodes
            v = self.route[current_edge_idx + 1]
            line = self.graph.G.edges[u, v]['geometry'] #Get the new edge

        valid_position = np.clip(remaining_dist, 0, line.length - 0.001) #Interpolate along the chosen edge
        pt = line.interpolate(valid_position) 
        
        next_pt = line.interpolate(min(valid_position + 1.0, line.length)) 
        future_angle = np.arctan2(next_pt.y - pt.y, next_pt.x - pt.x) #Calculate angle at this point for bounding box

        return self.get_bounding_box(x=pt.x, y=pt.y, angle=future_angle) #Return the bounding box representing the location in which the car will be
    

    def _get_projected_location(self, action, time):
        """
        Determines the projected location of this Vehicle after a certain period of time given a specific action
        """

        distance_travelled = self._calculate_projected_distance(action, time) #Calculate the projected distance
        return self._get_box_at_distance(distance_travelled) #Get the corresponding bounding box for this distance


    def get_possible_locations(self, dt):
        """
        Gets a representation of all the possible locations this Vehicle can be located, after a certain period of time and assuming maximum acceleration and braking constraints (no crashes)

        Uses _get_projected_location for both maximum acceleration and minimum acceleration, to determine a "snake-like" structure representing every possible location of the other car.

        Start of the snake: Location if this car slams its brakes on
        End of the snake: Location if this car floors the accelerator
        """

        current_d = self._calculate_projected_distance(VehicleAction.BRAKE, dt) #Start at brake distance
        dist_accel = self._calculate_projected_distance(VehicleAction.ACCEL, dt) #Eventually move up to accelerate distance

        boxes = []

        while current_d <= dist_accel: 
            box = self._get_box_at_distance(current_d) #Get the box at this distance
            boxes.append(box)
            
            current_d += self.length - 1 # Step size relative to car length

        if current_d < dist_accel + 4.0: #Ensure to capture last, max accel box to finish the snake
            boxes.append(self._get_box_at_distance(dist_accel))

        if boxes:
            return unary_union(boxes)
        else:
            return self.get_bounding_box() #Return this very box if nothing in boxes
        
    def _must_yield(self, other):
        """
        Returns True if 'self' must yield to 'other' based on traffic rules.
        """

        if self.current_start_node == other.current_start_node: #If on the same node
            if self.position > other.position: #If in front of them
                return False
            return True #If behind them
        
        if other.current_end_node == self.current_start_node: #Immediately return false if their end node is my start node
            return False
        
        if other.current_start_node == self.current_end_node: #Immediately return True if their start node is my end node
            return True

        if self.direction != Direction.STRAIGHT and other.direction == Direction.STRAIGHT: #If I am turning and the other guy is going straight, must yield
            return True
        
        if self.direction == Direction.LEFT and other.direction == Direction.RIGHT: #TODO This is adequate for now, if going LEFT and other is going RIGHT i must yield
            return True
            
        return False

    def action_decision(self, all_vehicles):
        """
        Will this car's projected location (based upon a specific action) intersect with another car's possible locations (action agnostic)? If so, then we brake. If not, then we accelerate

        Check the result for a multitude of timesteps, when the car is crossing the path.
        
        However, intersection bounding is a rather computationally expensive process (50 microseconds per, meaning 50 * 400 (20 car average) = 20ms or 50 fps which is atrocious for RL)

        For now, just do intersection bounding as we first want to get the simulation up and running, but if neccesary, generate a fixed number of large squares,
        a rough representation of the car bounding box, and check that inexpensively first. (AABB check)

        This implementation does not consider situations in which the car would accelreate, then brake, then accelerate and make it through the intersection that way. Since this is a small intersection.
        and calculating stuff like that is computationally expensive O(n^2), we keep to just accelerate checks for now. If ANY ONE of them fails, we stay braked.
        """

        t = 0.25
        while t < (self.max_speed / self.brake): #As long as the car can still brake in time
            my_future_box = self._get_projected_location(VehicleAction.ACCEL, t) #Calculate the acceleration box at this timestep

            for other in all_vehicles: #For every car (need to bounding box AABB to speed up TODO)
                if other.id == self.id: continue #If self dont care
                if other.done: continue #if done dont care
                if self._must_yield(other): #if i dont need to yield
                    other_future_box = other.get_possible_locations(t) #Get the projected location of this other car's possible actions box at this timestep
                else:
                    continue

                if my_future_box.intersects(other_future_box):  #If these two boxes intersect, immediately must brake.
                    return VehicleAction.BRAKE
            t += 0.25

        return VehicleAction.ACCEL #If none of these intersects, we can confirm the opportunity to go.


