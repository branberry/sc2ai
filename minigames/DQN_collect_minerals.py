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

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.linear_one = nn.Linear(7056,3528)
        self.linear_two = nn.Linear(3528, 21)
    
    def forward(self, observation):
        observation = F.relu(self.linear_one(observation))
        action_scores = self.linear_two(observation)
        return F.softmax(action_scores,dim=-1)

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optim.RMSprop(policy_net.parameters())
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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(21)]].view(1,1))

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
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
