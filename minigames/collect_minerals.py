from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features,units

import random


# Defining constants
PLAYER_SELF = features.PlayerRelative.SELF
PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
PLAYER_ENEMY = features.PlayerRelative.ENEMY


class MineralAgent(base_agent.BaseAgent):
    """
        This class defines a scripted agent to play the CollectMineralShards minigame.
    """
    def __init__(self):
        super(MineralAgent, self).__init__()

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]
        
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def coordinates(self, mask):
        """ 
            This method returns the x,y coordinates of a selected unit.
            Mask is a set of bools from comaprison with feature layer.
        """
        y,x = mask.nonzero()
        return list(zip(x,y))

    def step(self, obs):
        
        marines = self.get_units_by_type(obs, units.Terran.Marine)
        print(obs.observation.feature_screen.player_relative)
        if obs.first():
            return actions.FUNCTIONS.select_army("select")

        # checking if the desired function can be taken before we try using it! 
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            minerals = self.coordinates(player_relative == PLAYER_SELF)

            print(player_relative == PLAYER_SELF)
            return actions.FUNCTIONS.no_op()

