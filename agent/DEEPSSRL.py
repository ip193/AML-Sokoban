import torch

from agent.Agent import Agent
from agent.SSRL import SSRL, Episode, History
from agent.torchFFNN import FFNN


class DEEPSSRL(SSRL):
    """
    Implements multi-layered Stochastic Synapse Reinforcement Learning
    """
    def __init__(self, layers=(100, 100, 50, 4), nnet_bias=True, nonlinearity=torch.tanh, params=None, learning_rate=0.5,
                 decay=None):
        """
         Provide information about the neural network architecture and set up basic data structures
         :param layers: Layer sizes (first element is flattened input size, last element is flattened output size)
         :param nnet_bias: True if bias should be added to non-output layers
         :param nonlinearity: Nonlinearity function for use in the network
         :param params: If not None, use these parameters
         :param learning_rate: Learning rate for mean and standard deviations in formula
         :param decay: Rate of reward decay for computing updates (None => No decay)
         """

        super(SSRL).__init__()  # get Agent attributes

        self.layers = layers
        self.nnet_bias = nnet_bias
        self.nonlinearity = nonlinearity
        self.learning_rate = learning_rate

        self.default_name = "DEEPSSRL"
        self.resetName()

        self.NNET = FFNN(self.nonlinearity, self.nnet_bias)


        self.episode = Episode(self)
        self.params = params
        self.history = History()

        # TODO build torch params
