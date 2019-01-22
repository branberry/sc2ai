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
torch.set_default_tensor_type('torch.cuda.FloatTensor')

PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
PLAYER_SELF = features.PlayerRelative.SELF

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_TERRAN_MARINE = 48
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]

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
        self.dropout = nn.Dropout(.3)

        self.gamma = gamma
        self.state = []
        self.actions = []
        # Episode policy and reward history
        self.log_probs = []
        self.rewards = []

    def forward(self, observation):
        observation = F.relu(self.linear_one(observation))
        action_scores = self.linear_two(observation)
        return F.softmax(action_scores,dim=-1)


policy = VPG()

optimizer = optim.Adam(policy.parameters(), lr=1e-2) # utilizing the ADAM optimizer for gradient ascent
eps = np.finfo(np.float32).eps.item() # machine epsilon

def select_action(state):
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
    print(policy_loss)
    policy_loss = torch.stack(policy_loss,dim=0).sum()
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
        self.actions = []
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]
        
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def get_actions(self,feature_units,marine_coord):
        """
            This function returns a 2d-array 
            containing the coordinates of each mineral by order of
            closest mineral to furthest
        """
        coordinates = []
        for unit in feature_units:
            if unit[0] == 1680:
                dist = np.linalg.norm(np.array([unit[12],unit[13]]) - np.array(marine_coord))
                coords = [unit[12],unit[13]]
                coordinates.append([dist,coords])
        
        if len(coordinates) < 20:
            while len(coordinates) < 20:
                dist = np.linalg.norm(np.array([coordinates[0][1][0],coordinates[0][1][1]]) - np.array(marine_coord))
                coordinates.append([dist,[coordinates[0][1][0],coordinates[0][1][1]]])
        
        coordinates.sort(key=lambda x : x[0])
        res = []
        for coord in coordinates:
            res.append(coord[1])
        return res

    def step(self, obs):
        """
            method is called every frame
        """


        minerals = obs.observation['player'][1]
        if obs.last():
            finish_episode()
        if obs.first():
            self.step_minerals.append(minerals)

             #return actions.FUNCTIONS.
            unit_type = obs.observation.feature_screen[_UNIT_TYPE]
            marine_y,marine_x = (unit_type == _TERRAN_MARINE).nonzero()
            i = random.randint(0,len(marine_y) - 1)
            x = marine_x[i]
            y = marine_y[i]
            return actions.FunctionCall(_SELECT_POINT,[_NOT_QUEUED,[x,y]])
        
        res = obs.observation.feature_units
        while len(res) < 22:
            print(len(res))
            print(len([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
            res = np.append(res,[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],axis=0)
        #print(len(obs.observation.feature_minimap))
        input_data = torch.tensor(res)
        input_data = torch.flatten(input_data)
        input_data = input_data.float()
        print(input_data)
        #print(select_action(input_data))
        action = select_action(input_data)
        #policy_action = self.actions[action]

        
        unit_type = obs.observation.feature_screen[_UNIT_TYPE]
        marine_y,marine_x = (unit_type == _TERRAN_MARINE).nonzero()
        i = random.randint(0,len(marine_y) - 1)
        x = marine_x[i]
        y = marine_y[i]
        self.actions = self.get_actions(obs.observation.feature_units,[x,y])
        if minerals - self.step_minerals[len(self.step_minerals) - 1] > 0:
            reward = (minerals - self.step_minerals[len(self.step_minerals) - 1]) / 5
        else:
            reward = -1
        
        policy.rewards.append(reward)
        self.step_minerals.append(minerals)
        #print(self.actions[action])
        print(self.actions)
        #print(len(obs.observation.feature_units[0]))
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            return actions.FUNCTIONS.Move_screen("now",self.actions[action])
        else:
            return actions.FUNCTIONS.no_op()
