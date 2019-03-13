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

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
PLAYER_SELF = features.PlayerRelative.SELF

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

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

        self.gamma = gamma
        self.state = []
        self.actions = []

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(.50)

        # Neural network layers
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.linear_1 = nn.Linear(4624,120)
        self.linear_2 = nn.Linear(120,84)
        self.linear_3 = nn.Linear(84,21)


        # Episode policy and reward history
        self.log_probs = []
        self.rewards = []

    def forward(self, observation):
        observation = self.pool(F.relu(self.conv1(observation)))
        observation = self.pool(F.relu(self.conv2(observation)))
        observation = self.dropout(observation)
        print(observation)
        observation = observation.view(-1,4624)
        observation = F.relu(self.linear_1(observation))
        observation = F.relu(self.linear_2(observation))
        action_scores = self.linear_3(observation)
        return action_scores


# Instantiating the neural network that will serve as the policy gradient 
policy = VPG()

policy.cuda()

optimizer = optim.Adam(policy.parameters(), lr=1e-1) # utilizing the ADAM optimizer for gradient ascent
eps = np.finfo(np.float32).eps.item() # machine epsilon

def select_action(state,steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.* steps_done / EPS_DECAY)

    probs = policy(state)
    # creates a categorical distribution
    # a categorical distribution is a discrete probability distribution that describes 
    # the possible results of a random variable.  In this case, our possible results are our available actions
    m = Categorical(probs) 
    if sample > eps_threshold:
        action = m.sample() 
        print("action from policy " + str(action))
        policy.log_probs.append(m.log_prob(action))
        return action.item()
    else:
        action = torch.tensor(random.randrange(21),device=device, dtype=torch.long)
        policy.log_probs.append(m.log_prob(action))
        print("Random action: " + str(action))
        return action


def finish_episode():
    """
        This method is called at the end of the episode to compute the policy loss i.e. to see how 
        well the model performed, and then upgrade the neural network based on this metric.

    """
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
    """

    """
    def __init__(self):
        super(SmartMineralAgent, self).__init__()
        self.step_minerals = []
        self.reward = 0
        self.episode_count = 0
        self.actions = []
        self.start_data = []
        self.steps = 0


    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]
        
    def can_do(self, obs, action):
        return action in obs.observation.available_actions
    

    def get_actions(self,feature_units):
        """
            This function returns a 2d-array 
            containing the coordinates of each mineral by order of
            closest mineral to furthest
        """
        coordinates = []
        for unit in feature_units:
            if unit[0] == 1680:
                dist = np.linalg.norm(np.array([unit[12],unit[13]]))
                coords = [unit[12],unit[13]]
                coordinates.append([dist,coords])
  
        res = []

        for coord in coordinates:
            res.append(coord[1])

        while len(res) < 21:
            res.append([999,999])
        
        print(res)
        return res

    def step(self, obs):
        """
            The step method is the meat and bones of our experiment.  
            In the step function,we have accesss to all of the game data within the obs object.
            This method is called every frame of the game, and it MUST return an action function.
        """

        # grabbing the current mineral count at the given frame
        minerals = obs.observation['player'][1] 


        # 
        if obs.first():
            self.episode_count += 1
            # append the start mineral count
            # we use the mineral count to determine the 
            self.step_minerals.append(minerals)
            self.start_data = obs.observation.feature_units
            player_relative = obs.observation.feature_screen.player_relative
            marines = coordinates(player_relative == PLAYER_SELF)
            self.actions = self.get_actions(obs.observation.feature_units)
            marine_coordinates = np.mean(marines, axis=0).round()  # Average location.
            return actions.FUNCTIONS.select_army("select")

        # obs.last() returns a boolean if the frame is the last in an episode or not
        if obs.last():
            print("Episode " + str(self.episode_count) + " completed")
            #self.steps = 0
            finish_episode()


        input_data = torch.tensor([[obs.observation.feature_screen[6],obs.observation.feature_screen[4],obs.observation.feature_screen[5]]]).float()
        
        player_relative = obs.observation.feature_screen.player_relative

        #state_preprocess(obs.observation.feature_screen[4])

        # obs.observation.feature_screen[4] represents the player_id screen of the pysc2 GUI
        marines = coordinates(player_relative == PLAYER_SELF)


        marine_coordinates = np.mean(marines, axis=0).round()  # Average location.


        if minerals - self.step_minerals[len(self.step_minerals) - 1] > 0:

            # based on the number of minerals collected from a given step,
            # we want to reward the agent accordingly. So, if the agent collected
            # two mineral shards, the difference would be:
            # (200 - 0) // 100 = 2 (integer division gives us nice whole numbers)
            # The self.step_minerals array contains the previous minerals from all previous steps
            # and the minerals variable contains the current mineral count for the agent 
            self.reward += 1
        else:
            self.reward += -1



        policy.rewards.append(self.reward)
        self.step_minerals.append(minerals)
        #self.actions = self.get_actions(marine_coordinates)
        # return the action that the policy chose!
        self.steps += 1

        action = select_action(input_data,self.steps)
        
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            if self.actions[action][0] != 999:
                return actions.FUNCTIONS.Move_screen("now",self.actions[action])
            else:
                return actions.FUNCTIONS.no_op()
        else:
            return actions.FUNCTIONS.no_op()
