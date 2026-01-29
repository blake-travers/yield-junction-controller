import os
import sys
import traci
import sumolib
import numpy as np

DIR_LEFT     = [1, 0, 0]
DIR_STRAIGHT = [0, 1, 0]
DIR_RIGHT    = [0, 0, 1]

class Vehicle:
    """
    Wrapper to extract and clean data from the environment and import it cleanly into the GNN
    """
    def __init__(self, vehicleID):
        self.id = vehicleID
        self.position = 0.0
        self.speed = 0.0
        self.waittime = 0.0
        self.max_speed = 15 #traci.vehicle.getMaxSpeed(self.id)

        traci.vehicle.setLaneChangeMode(self.id, 0) #Disable all lane changing for this vehicle

        self.direction = self._getStaticDirection()
        self.lane_index = -1 #Represents a value between -1 and 1 (where 0 is the intersection) on how far through the route we are
        self.lane_id = "" #Lane ID used to calculate length
        self.lane_length = 100
        self.done = False
        self.update()

    def update(self):
        if not self.done:
            try:
                self.position = traci.vehicle.getLanePosition(self.id)
                self.speed = traci.vehicle.getSpeed(self.id)
                self.wait_time = traci.vehicle.getAccumulatedWaitingTime(self.id)
                new_lane_id = traci.vehicle.getLaneID(self.id)

                if new_lane_id == "":
                    return
                elif new_lane_id != self.lane_id: #If we've just changed edges, i.e. different id
                    self.lane_id = new_lane_id #Set this to the new index
                    self.lane_index = traci.vehicle.getRouteIndex(self.id) - 1 #Because we want between -1 and 1.. this has to change if route length becomes more than 3 in the future
                    self.lane_length = traci.lane.getLength(self.lane_id) #Get the new lane length

            except traci.TraCIException:
                self.done = True
        
    def _getStaticDirection(self):
        """
        Utilises traci.vehicle.getRoute(self.id) to determine the direction this vehicle will travel
        """

        try:
            route = traci.vehicle.getRoute(self.id)

            compass = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
            start = compass[route[0][-1]] #Get the input lane's direction
            end   = compass[route[1][-1]] #Get the output lane's direction

            diff = (end - start) % 4 #Determine the difference, representing the direction

            if diff == 3:
                return DIR_LEFT
            elif diff == 1:
                return DIR_RIGHT
            elif diff == 2:
                return DIR_STRAIGHT
            else:
                raise ValueError("Direction Cannot be Calculated. Potential U-turn detected")

        except:
            raise ValueError(f"Direction Cannot be Calculated for vehicle {self.id}")
        

    def toVector(self):
        """
        Converts the information from each vehicle traci class into useable information in the GNN
        
        """

        normalised_position = max(0.0, min(self.position / self.lane_length, 1.0))
        normalised_speed = min(self.speed / self.max_speed, 1.0)
        normalised_wait_time = self.wait_time/60

        return [normalised_position, normalised_speed] + self.direction

    def getLaneID(self):
        return self.lane_id
    
    def getPosition(self):
        return self.position #TODO this could be changed to give more informative information to the agent in future
    
    def getDirection(self):
        return self.direction
