
import numpy as np

import agent
from agent.FFNN import FFNN


class Params:
    """
    Hold parameters for the agent here.
    """

    def __init__(self, layers, nnet_bias=True):
        """
        :param layers: List of layer sizes (not including bias)
        """
        self.layers = layers
        self.nnet_bias = nnet_bias

        self.means = []
        self.stds = []

    def checkParams(self):
        """
        Checks to see if each layer has the correct number of parameters.
        :return:
        """
        error = False

        if self.means[0].shape != self.stds[0].shape:
            error = True

        for i in range(1, len(self.layers)):
            if self.means[i].shape != self.stds[i].shape:
                error = True

            if i != len(self.layers) - 1:
                if self.means[i].shape[1] != self.means[i - 1].shape[0] + int(self.nnet_bias):
                    error = True

                if self.means[i].shape[1] != self.layers[i - 1] + int(self.nnet_bias):
                    error = True

            else:
                if self.layers[i] != self.means[i].shape[0]:
                    error = True

        if error:
            raise RuntimeError("Param object with incorrect number of parameters.")

    def setMeans(self, means):
        self.means = means

    def setStds(self, stds):
        self.stds = stds

    def initializeRandom(self, random_bounds = ((-1, 1) , (0.05, 1))):
        """
        Set up parameters uniformly randomly.
        Note: If bias nodes are used, this function will add parameters accordingly.
        :param random_bounds: Bounds of means and variances for random setup
        :return:
        """

        bias = int(self.nnet_bias)

        for i, size in enumerate(self.layers):
            if i == 0:
                continue

            self.means.append(np.random.rand(size, self.layers[i-1] + bias)*(random_bounds[0][1] - random_bounds[0][0])
                              + random_bounds[0][0])

            self.stds.append(np.random.rand(size, self.layers[i-1] + bias)*(random_bounds[1][1] - random_bounds[1][0])
                              + random_bounds[1][0])


class SSRL(agent.Agent):
    """
    Implements Stochastic Synapse Reinforcement Learning
    """

    def __init__(self, layers, nnet_bias, params = None):

        """
        Provide information about the neural network architecture and set up basic data structures
        :param layers: Layer sizes (first element is flattened input size, last element is flattened output size)
        :param nnet_bias: True if bias should be added to non-output layers
        :param params: If not None, use these parameters
        """

        super().__init__()
        self.layers = layers
        self.nnet_bias = nnet_bias
        self.rewards = None
        self.means_eligibility_traces, self.stds_eligibility_traces = None, None

        if params is not None:  # check if dimensions are correct

            params.checkParams()
            self.setParams()

    def resetEpisode(self, episode_tag=None):  # TODO: Add tag
        """
        Reset the episode
        :param episode_tag: Episode identifier
        :return:
        """

        self.rewards = []

        self.timeStep = 0
        self.episode_tag = episode_tag  # parent class attributes

        self.means_eligibility_traces = [np.zeros(m.shape) for m in self.params.means]
        self.stds_eligibility_traces = [np.zeros(s.shape) for s in self.params.stds]

    def setParams(self, params = None):
        """
        Set up parameter attributes (generate randomly unless otherwise specified)
        :param p: Params object or None
        :return:
        """

        if params is not None:
            self.params = params
            return

        else:
            self.params = Params(self.layers, self.nnet_bias)
            self.params.initializeRandom()

    





