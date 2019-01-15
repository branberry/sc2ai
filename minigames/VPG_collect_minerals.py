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


_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
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
        self.linear_two = nn.Linear(1144, 25)

        self.gamma = gamma
        self.state = []
        self.actions = []
        # Episode policy and reward history
        self.log_probs = []
        self.rewards = []

    def forward(self, observation):
        observation = F.relu(self.linear_one(observation))
        action_scores = self.linear_two(observation)
        return F.softmax(action_scores,dim=1)


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


class SmartMineralAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(SmartMineralAgent, self).__init__()
        self.step_minerals = []
        self.reward = 0

    def coordinates(self, mask):
        """ 
            This method returns the x,y coordinates of a selected unit.
            Mask is a set of bools from comaprison with feature layer.
        """
        y,x = mask.nonzero()
        return list(zip(x,y))


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

        player_relative = obs.observation.feature_screen.player_relative 

        mineral_count = obs.observation['player'][1]
        
        if obs.first():
            self.step_minerals.append(mineral_count)
            mineral_coordinates = self.coordinates(player_relative == PLAYER_NEUTRAL)
            policy.actions = mineral_coordinates
            print(mineral_coordinates)

            return actions.FUNCTIONS.select_army("select")
 
        else:
            if mineral_count - self.step_minerals[len(self.step_minerals) - 1] > 0:
                reward = (mineral_count - self.step_minerals[len(self.step_minerals) - 1]) / 5
            else:
                reward = -1
            
            policy.rewards.append(reward)
            self.step_minerals.append(mineral_count)

        #state = obs.observation.feature_units 

        marines = self.get_units_by_type(obs, units.Terran.Marine)
        
        if ACTION_ATTACK_MINIMAP in obs.observation['available_actions']:
            pass 

        return actions.FUNCTIONS.no_op()