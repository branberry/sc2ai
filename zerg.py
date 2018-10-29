from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app


class ZergAgent(base_agent.BaseAgent):
    """
        The zerg agent class will be the game agent 
        for our simulations
    """
    def step(self, obs):
        """
            The step method takes in an observation of the 
            game world, and then returns an action for the 
            agent
        """
        super(ZergAgent, self).step(obs)
        
        return actions.FUNCTIONS.no_op()