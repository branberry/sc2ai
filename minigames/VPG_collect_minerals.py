"""
    TODO: We will be using the feature_units of the observation as input into our neural network.
    The output of our neural network will be the action to move to a crystal, and since there are 20 crystals, we will 
    have 20 output neurons.  The output neurons will be ordered by minerals closest to the average distance between the marines.
"""

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features,units

import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
PLAYER_SELF = features.PlayerRelative.SELF

ACTION_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

ACTION_DO_NOTHING = 0
ACTION_MOVE_CAMERA = 1
ACTION_SELECT_POINT = 2
ACTION_SELECT_RECT = 3
ACTION_SELECT_GROUP = 4
ACTION_SELECT_ARMY = 7
ACTION_ATTACK_SCREEN = 12

GAMMA = 0.99

class VPG(nn.Module):
    def __init__(self,gamma=0.99):
        super(VPG, self).__init__()

        self.linear_one = nn.Linear(572,1144)
        self.linear_two = nn.Linear(1144, 20)

        self.gamma = gamma
        self.state = []
        self.actions = []
        # Episode policy and reward history
        self.log_probs = []
        self.rewards = []

    def forward(self, observation):
        observation = F.relu(self.linear_one(observation))
        action_scores = self.linear_two(observation)
        return F.relu(action_scores,dim=1)


policy = VPG()

optimizer = optim.Adam(policy.parameters(), lr=1e-2) # utilizing the ADAM optimizer for gradient ascent
eps = np.finfo(np.float32).eps.item() # machine epsilon

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0) # retreiving the current state of the game to determine a action
    probs = policy(state)
    # creates a categorical distribution
    # a categorical distribution is a discrete probability distribution that describes 
    # the possible results of a random variable.  In this case, our possible results are our available actions
    m = Categorical(probs) 
    action = m.sample()
    policy.log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards:
        R = r + GAMMA * R
        rewards.insert(0,R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for log_prob, reward in zip(policy.log_probs, rewards):
        policy_loss.append(-log_prob*reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.log_probs[:]

def coordinates(mask):
    """ 
        This method returns the x,y coordinates of a selected unit.
        Mask is a set of bools from comaprison with feature layer.
    """
    y,x = mask.nonzero()
    return list(zip(x,y))
class SmartMineralAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(SmartMineralAgent, self).__init__()
        self.step_minerals = []
        self.reward = 0




    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]
        
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def step(self, obs):
        """
            method is called every frame
        """
        if obs.last():
            #finish_episode()
            self.reward = 0

        player_relative = obs.observation.feature_screen.player_relative 

        mineral_count = obs.observation['player'][1]
        
        if obs.first():
            self.step_minerals.append(mineral_count)
            mineral_coordinates = coordinates(player_relative == PLAYER_NEUTRAL)
            policy.actions = mineral_coordinates
            #print(len(mineral_coordinates))
            #print(obs.observation['available_actions'])
            #marines = coordinates(player_relative == PLAYER_SELF)
            #print(marines)

            feature_units = np.array(obs.observation.feature_units)
            feature_minimap = np.array(obs.observation.feature_minimap)
            feature_screen = np.array(obs.observation.feature_screen)

            
            
            print(obs.observation.feature_units) # this will be the value for the number of input neurons
            
            return actions.FUNCTIONS.select_army("select")

        

        self.reward += obs.reward

        
        policy.rewards.append(self.reward)
        self.step_minerals.append(mineral_count)

        #state = obs.observation.feature_units 

        
        
        if ACTION_ATTACK_MINIMAP in obs.observation['available_actions']:
            pass 

        return actions.FUNCTIONS.no_op()