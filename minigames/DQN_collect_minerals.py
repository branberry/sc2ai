from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features,units

import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNMineralAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(DQNMineralAgent, self).__init__()

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]
        
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        """
            method is called every frame
        """
        marines = self.get_units_by_type(obs, units.Terran.Marine)
        
        print(obs)
        if obs.first():
            return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.Attack_minimap("now",
                                                (random.randint(0,50),random.randint(0,50)))
