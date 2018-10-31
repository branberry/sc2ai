from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random


class ZergAgent(base_agent.BaseAgent):
    """
        The zerg agent class will be the game age
        for our simulations
    """
    def step(self, obs):
        """
            The step method takes in an observation of the 
            game world, and then returns an action for the 
            agent
        """
        super(ZergAgent, self).step(obs)

        larvae = self.get_units_by_type(obs,units.Zerg.Larva)
        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)

        if self.unit_type_is_selected(obs,units.Zerg.Larva):
            if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("now")
        if len(larvae) > 0:
            # checks for larvae, and selects all 
            larva = random.choice(larvae)
            
            return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))
        # checks if the unit we are looking at is a drone
        if self.unit_type_is_selected(obs, units.Zerg.Drone) and len(spawning_pools) == 0:
            if ( self.can_do(obs,actions.FUNCTIONS.Build_SpawningPool_screen.id)):
                x = random.randint(0,83)
                y = random.randint(0,83)

                return actions.FUNCTIONS.Build_SpawningPool_screen("now",(x,y))

        drones = self.get_units_by_type(obs, units.Zerg.Drone)

        if len(drones) > 0:
            drone = random.choice(drones)

            return actions.FUNCTIONS.select_point("select_all_type", (drone.x,drone.y))

        return actions.FUNCTIONS.no_op()
    
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
            return True
        
        if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

    def can_do(self, obs, action):
        """ This method is used to simplify checking if an action is available """
        return action in obs.observation.available_actions