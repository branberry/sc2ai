import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class VPG(nn.Module):

    def __init__(self):
        super(VPG, self).__init__()