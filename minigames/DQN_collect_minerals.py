from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features,units
from collections import namedtuple

import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
PLAYER_SELF = features.PlayerRelative.SELF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Saves transition"""

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Hyper Parameters

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)



steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.* steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value in each row
            return policy_net(state).max(0)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(21)]],device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t,a).  The model computes Q(s_t), then we select the columsn
    # of the action taken.  These are the actions which would ahve been taken for
    # each state within the batch according to the policy
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    print(state_action_values)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def coordinates(mask):
    """ 
        This method returns the x,y coordinates of a selected unit.
        Mask is a set of bools from comaprison with feature layer.
    """
    y,x = mask.nonzero()
    return list(zip(x,y))
class DQNMineralAgent(base_agent.BaseAgent):
    
    def __init__(self):
        super(DQNMineralAgent, self).__init__()
        self.step_minerals = []
        self.reward = 0
        self.episode_count = 0
        self.actions = []

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]
        
    def can_do(self, obs, action):
        return action in obs.observation.available_actions
    
    def get_actions(self,marine_coord,feature_units):
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
  
        coordinates.sort(key=lambda x : x[0])
        res = []

        for coord in coordinates:
            res.append(coord[1])

        while len(res) < 21:
            res.append([999,999])
        
        #print(res)
        return res

    def step(self, obs):
        """
            method is called every frame
        """
        # grabbing the current mineral count at the given frame
        minerals = obs.observation['player'][1] 
        if obs.first():
            self.step_minerals.append(minerals)
            player_relative = obs.observation.feature_screen.player_relative
            marines = coordinates(player_relative == PLAYER_SELF)
            return actions.FUNCTIONS.select_army("select")
        
        player_relative = obs.observation.feature_screen.player_relative

        marines = coordinates(player_relative == PLAYER_SELF)
        
        marine_coordinates = np.mean(marines, axis=0).round()  # Average location.

        self.actions = self.get_actions(marine_coordinates,obs.observation.feature_units)



        if minerals - self.step_minerals[len(self.step_minerals) - 1] > 0:

            # based on the number of minerals collected from a given step,
            # we want to reward the agent accordingly. So, if the agent collected
            # two mineral shards, the difference would be:
            # (200 - 0) // 100 = 2 (integer division gives us nice whole numbers)
            # The self.step_minerals array contains the previous minerals from all previous steps
            # and the minerals variable contains the current mineral count for the agent 
            self.reward += (minerals - self.step_minerals[len(self.step_minerals) - 1])/100
        else:
            self.reward += -1.0


        
        #state = torch.tensor(obs.observation.feature_screen[4]).flatten().float()
        state = torch.tensor(obs.observation.feature_screen[4])
        action = select_action(state) 

        if not obs.last():
            next_state = torch.tensor(obs.observation.feature_screen[4]).flatten().float()
        else:
            next_state = None
        
        reward = torch.tensor([self.reward], device=device)

        memory.push(state, action, next_state, reward)

        self.step_minerals.append(minerals)

        optimize_model()


        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            if self.actions[action][0] != 999:
                return actions.FUNCTIONS.Move_screen("now",self.actions[action])
            else:
                return actions.FUNCTIONS.no_op()
        else:
            return actions.FUNCTIONS.no_op()
