"""
    This file contains the implementation of the Q-Learning algorithm
    I will be adapting the code from here: https://github.com/skjb/pysc2-tutorial/blob/master/Building%20a%20Smart%20Agent/smart_agent_step5.py
    to work with the Zerg race as opposed to the terrans.
"""

import random
import math

import torch
import numpy as np
import pandas as pd

from pysc2.lib import actions
from pysc2.lib import features


class QLearn:
    def __init__(self,actions,learning_rate=0.01, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma # the reward decay or 'discount'
        self.epsilon = epsilon # the initial percentage to take a random action
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)