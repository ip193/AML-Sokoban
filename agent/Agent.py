
from abc import ABCMeta, abstractmethod
import numpy as np

class Agent(metaclass=ABCMeta):
    """
    Agent interface to handle action output, rewards, etc.
    """

    def __init__(self):
        pass

    @abstractmethod
    def resetEpisode(self, episode_tag):
        """
        Clear buffers/running values, rename episode, and reset time step to 0.
        :param episode_tag: New episode name
        :return:
        """
        pass

    @abstractmethod
    def incrementTimeStep(self):
        """
        Increase the time step in this episode by one.
        :return:
        """
        pass

    @abstractmethod
    def giveObservation(self, obs):
        """
        Provide the observation to the agent.
        :param obs: Observation of the world (numpy array)
        :return:
        """
        pass

    @abstractmethod
    def act(self):
        """
        Return the action label (after having received the observation).
        :return: Action label
        """
        pass

    @abstractmethod
    def giveReward(self, reward):
        """
        Provide the reward to the agent for the last action taken.
        :return:
        """
        pass

    @abstractmethod
    def endOfEpisodeUpdate(self):
        """
        Update parameters after the conclusion of an episode.
        :return:
        """
        pass

    @abstractmethod
    def saveModel(self, filename, folder=None):  #  TODO should this be abstract? Folder saving conventions?
        """
        Save the model to the disk.
        :param filename: Filename to save to
        :param folder:
        :return:
        """
        pass