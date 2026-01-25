import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.special import comb
from itertools import combinations
from enum import Enum

class Direction(Enum):
    LEFT = "LEFT"
    STRAIGHT = "STRAIGHT"
    RIGHT = "RIGHT"

#TODO: Switch Nodes to Enums

class IntersectionGraph:
    def __init__(self):
        self.G = nx.DiGraph()

        QUEUE_LENGTH = 15 #Decides the length of the queue adjacent to both the incoming and outgoing nodes of the intersection

        INTERSECTION_WIDTH = 12.5 #Horizontal half-size of the intersection
        INTERSECTION_HEIGHT = 9 #Vertical half-size of the intersection
        LANE_WIDTH = 3.5 #Diameter of each Lane. Cars will not completely fill up each lane, being approximately 60% of the size

        LANE_GAP = 0.5 #Represents the median in the middle, or the gap, between incoming and outgoing lanes on the same road. Two incoming lanes are assumed to have no gap.

        self.curve_points = 10

        #Centre is 0,0
        self.in_nodes = { #In Nodes represent nodes in which existing cars will use to ENTER the intersection
            "inN1": (0.5*LANE_GAP + 1.5*LANE_WIDTH, INTERSECTION_HEIGHT), #N1 and S1 always refer to the leftmost lane (turning left into E / W)
            "inN2": (0.5*LANE_GAP + 0.5*LANE_WIDTH, INTERSECTION_HEIGHT),
            "inS1": (-0.5*LANE_GAP - 1.5*LANE_WIDTH, -INTERSECTION_HEIGHT),
            "inS2": (-0.5*LANE_GAP - 0.5*LANE_WIDTH, -INTERSECTION_HEIGHT),
            "inE": (INTERSECTION_WIDTH, -0.5*LANE_GAP - 0.5*LANE_WIDTH),
            "inW": (-INTERSECTION_WIDTH, 0.5*LANE_GAP + 0.5*LANE_WIDTH)
        }

        self.out_nodes = { #Out Nodes represent nodes in which existing cars will use to EXIT the intersection
            "outN1": (-0.5*LANE_GAP - 1.5*LANE_WIDTH, INTERSECTION_HEIGHT), #N1 and S2 in this case refer to the rightmost lane, oppisite to the in nodes
            "outN2": (-0.5*LANE_GAP - 0.5*LANE_WIDTH, INTERSECTION_HEIGHT), #This means N for example will look like this: [outN1   outN2    inN2   inN1]
            "outS1": (0.5*LANE_GAP + 1.5*LANE_WIDTH, -INTERSECTION_HEIGHT),
            "outS2": (0.5*LANE_GAP + 0.5*LANE_WIDTH, -INTERSECTION_HEIGHT),
            "outE": (INTERSECTION_WIDTH, 0.5*LANE_GAP + 0.5*LANE_WIDTH),
            "outW": (-INTERSECTION_WIDTH, -0.5*LANE_GAP - 0.5*LANE_WIDTH)
        }

        self.in_queue = { #Nodes connected to in_nodes, providing the agent with information about the queue
            "inqN1": (self.in_nodes["inN1"][0], self.in_nodes["inN1"][1] + QUEUE_LENGTH),
            "inqN2": (self.in_nodes["inN2"][0], self.in_nodes["inN2"][1] + QUEUE_LENGTH),
            "inqS1": (self.in_nodes["inS1"][0], self.in_nodes["inS1"][1] - QUEUE_LENGTH),
            "inqS2": (self.in_nodes["inS2"][0], self.in_nodes["inS2"][1] - QUEUE_LENGTH),
            "inqE":  (self.in_nodes["inE"][0] + QUEUE_LENGTH, self.in_nodes["inE"][1]),
            "inqW":  (self.in_nodes["inW"][0] - QUEUE_LENGTH, self.in_nodes["inW"][1]),
        }

        self.out_queue = { #Nodes connected to out_nodes, representing the end of the road (potential bottleneck capacity)
            "outqN1": (self.out_nodes["outN1"][0], self.out_nodes["outN1"][1] + QUEUE_LENGTH),
            "outqN2": (self.out_nodes["outN2"][0], self.out_nodes["outN2"][1] + QUEUE_LENGTH),
            "outqS1": (self.out_nodes["outS1"][0], self.out_nodes["outS1"][1] - QUEUE_LENGTH),
            "outqS2": (self.out_nodes["outS2"][0], self.out_nodes["outS2"][1] - QUEUE_LENGTH),
            "outqE":  (self.out_nodes["outE"][0] + QUEUE_LENGTH, self.out_nodes["outE"][1]),
            "outqW":  (self.out_nodes["outW"][0] - QUEUE_LENGTH, self.out_nodes["outW"][1]),
        }
    
        self.intersection_nodes = {**self.in_nodes, **self.out_nodes}
        self.queue_nodes = {**self.in_queue, **self.out_queue}
        self.all_nodes = {**self.intersection_nodes, **self.queue_nodes}

        for name, pos in self.all_nodes.items():
            self.G.add_node(name, pos=pos)

    def add_edge(self, start_id, end_id, turn_type="STRAIGHT"):
        p1 = np.array(self.all_nodes[start_id]) #Get Start and End Position
        p2 = np.array(self.all_nodes[end_id])

        if turn_type == "STRAIGHT":
            t = np.linspace(0, 1, self.curve_points)
            curve_points = p1 + t[:, None] * (p2 - p1) #Simple straight line of self.curve_points size
            
        else: #Must be a turn, create a beizer curve representing the curve cars will take
            v1 = self._direction_from_name(start_id) #Get start and end directions, to determine the orientation of the curve
            v2 = self._direction_from_name(end_id)

            intersect = self._get_line_intersection(p1, v1, p2, -v2) #Find the coordinates two lines intersect, such that a curve can be generated off those coordinates
            
            if intersect is not None:
                points = [p1, intersect, p2] #Concatanate a list of three points representing the beizer curve
            else: #Fallback just in case lines are parellel (in cases such as a U-turn which are not supported yet)
                mid = (p1 + p2) / 2
                points = [p1, mid, p2]

            xs, ys = self._bezier_curve(points) #Generate a beizer curve with n number of points
            curve_points = np.column_stack([xs, ys]) #Convert these into a list of points
        
        geom = LineString(curve_points)
        self.G.add_edge(start_id, end_id, geometry=geom, turn_type=turn_type)

    def _direction_from_name(self, name):
        """
        Infers the driving direction vector based on the node name, used to calculate polynomial direction.
        """
        if "inN" in name or "outS" in name: return np.array([0, -1])  # Down
        if "outN" in name or "inS" in name: return np.array([0, 1])  # Up
        if "inE" in name or "outW" in name: return np.array([-1, 0])  # Left
        if "outE" in name or "inW" in name: return np.array([1, 0])  # Right
        return np.array([0, 0])

    def _bernstein_poly(self, i, n, t): #Helper
        return comb(n, i) * ( t**i ) * (1 - t)**(n-i)

    def _bezier_curve(self, points):
        """
        Generate an Eliptical curve based upon the three points calculated previously
        """
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        t = np.linspace(0.0, 1.0, self.curve_points)
        
        polynomial_array = np.array([self._bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def _get_line_intersection(self, p1, v1, p2, v2):
        """
        Get the coordinates of the place two lines will intersect. Does not work properly for parallel lines obviously
        """
        det = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(det) < 1e-5: return None

        diff = p2 - p1
        t = (diff[0] * v2[1] - diff[1] * v2[0]) / det
        return p1 + t * v1


    def build_lanes(self):
        """
        Creates lanes for all queue nodes and their corresponding intersection node (Directional)
        Creates lanes for every edge required inside the intersection

        Highly dependant on current Queue naming conventions
        """
        print("Building [inq -> in] edges...")
        for node_name in self.in_nodes: #For each of the in nodes
            queue_name = "inq" + node_name.replace("in", "") # Find its corresponding queue node
            if queue_name in self.in_queue: #Not neccesary, but make sure that it is inside the correct queue
                self.add_edge(queue_name, node_name, "STRAIGHT") #Add a straight edge between these

        print("Building [out -> outq] edges...")
        for node_name in self.out_nodes: #Same for out queue nodes
            queue_name = "outq" + node_name.replace("out", "")
            if queue_name in self.out_queue:
                self.add_edge(node_name, queue_name, "STRAIGHT")

        print("Building [in -> out] edges...")
        lanes = [
            ("inN1", "outE", "LEFT"),       #Left turn from North side
            ("inN1", "outS1", "STRAIGHT"),  #Straight from North side, using left lane
            ("inN2", "outS2", "STRAIGHT"),  #Straight from North side, using right lane
            ("inN2", "outW", "RIGHT"),      #Right turn from North side

            ("inS1", "outW", "LEFT"),       #Left turn from South side
            ("inS1", "outN1", "STRAIGHT"),  #Straight from South side, using left lane
            ("inS2", "outN2", "STRAIGHT"),  #Straight from North side, using right lane
            ("inS2", "outE", "RIGHT"),      #Right turn from South side

            ("inE", "outS1", "LEFT"),       #Left turn from East side
            ("inE", "outW", "STRAIGHT"),    #Straight from East side
            ("inE", "outN2", "RIGHT"),      #Right turn from East sidde

            ("inW", "outN1", "LEFT"),       #Left turn from West side
            ("inW", "outE", "STRAIGHT"),    #Straight from West side
            ("inW", "outS2", "RIGHT"),      #Right turn from West side
        ]
        
        for start, end, t in lanes:
            if start in self.intersection_nodes and end in self.intersection_nodes:
                self.add_edge(start, end, t)
            else:
                print(f"Warning: Could not connect {start} to {end} (Node not found)")
                


    def visualize(self):
        plt.figure(figsize=(10, 10))
        
        #Draw Nodes with specific colors
        for n, pos in self.G.nodes(data='pos'):
            color = 'black'
            label_offset = 1
            
            if "inq" in n:     color = 'green'
            elif "outq" in n:  color = 'gray'
            elif "in" in n:    color = 'red'
            elif "out" in n:   color = 'blue'
            
            plt.plot(pos[0], pos[1], 'o', color=color, markersize=6)
            plt.text(pos[0]+label_offset, pos[1]+label_offset, n, fontsize=8) #Text with offset

        # 2. Draw Edges with Arrows
        for u, v, data in self.G.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                
                # Color Edges based on Turn Type
                edge_color = 'gray'
                if data.get('turn_type') == 'LEFT': edge_color = '#3498db'   # Blue-ish
                if data.get('turn_type') == 'RIGHT': edge_color = '#e67e22'  # Orange-ish
                
                # Plot the line
                plt.plot(xs, ys, color=edge_color, alpha=0.6, linewidth=2)
                
                # Draw a direction arrow at the midpoint
                mid = len(xs) // 2
                # Use a small vector between mid and mid+1 to orient the arrow
                dx = xs[mid+1] - xs[mid]
                dy = ys[mid+1] - ys[mid]
                
                plt.arrow(xs[mid], ys[mid], dx, dy, 
                          shape='full', lw=0, length_includes_head=True, head_width=0.8, color=edge_color)

        plt.grid(True)
        plt.title("Traffic Graph: Green=Spawn, Red=StopLine, Blue=Exit")
        plt.axis('equal') # Important! Keeps the aspect ratio 1:1 so turns look real
        plt.show()

if __name__ == "__main__":
    graph = IntersectionGraph()
    graph.build_lanes()
    graph.visualize()
