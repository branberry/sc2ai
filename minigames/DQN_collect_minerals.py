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

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()


model = DQN()

model.cuda()
def optimize_model():
    pass

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.* steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:

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
        
        if obs.first():
            return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.Attack_minimap("now",
                                                (random.randint(0,50),random.randint(0,50)))
