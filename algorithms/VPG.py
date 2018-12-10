import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

<<<<<<< HEAD
class VPG(nn.Module):

    def __init__(self):
        super(VPG, self).__init__()
=======
SEED = 543

env = gym.make('CartPole-v0')
env.seed(SEED)
>>>>>>> b65c3c6487e63680855e7082d46149ca1a88b4e0
