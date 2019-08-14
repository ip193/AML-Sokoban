
from abc import ABCMeta, abstractmethod
import numpy as np

class Agent(metaclass=ABCMeta):
    """
    Agent interface to handle action output, rewards, etc.
    """
    @abstractmethod
    def getAction(self):
        pass

    @abstractmethod
    def giveReward(self):
        pass

    