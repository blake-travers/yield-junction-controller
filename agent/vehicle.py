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

        self.direction = self._get_static_direction()
        self.done = False
        self.update()

    def update(self):
        if not self.done:
            try:
                self.position = traci.vehicle.getLanePosition(self.id)
                self.speed = traci.vehicle.getSpeed(self.id)
                self.wait_time = traci.vehicle.getAccumulatedWaitingTime(self.id)  
            except traci.TraCIException:
                self.done = True
        
    def _get_static_direction(self):
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
