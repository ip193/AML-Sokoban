


import numpy as np


class FFNN():
    """
    Implementation of a simple feed-forward neural net
    """

    def __init__(self, params, nonlinearity, bias=True):

        """
        :param params: List of layer weights
        :param nonlinearity: function handle for activation function
        :param bias: True if a bias node should be added to each layer
        """

        self.params = params
        self.nl = nonlinearity
        self.bias = bias

    def forward(self, input):

        input = input.flatten()
        if input.shape[0] + int(self.bias) != self.params[0].shape[1]:
            raise RuntimeError("Incorrect input dimensions. Got: "+input.shape + " Expected: "+self.params[0].shape)

        activation = input # holds the activation of the previous layer

        for i, p in enumerate(self.params):

            if self.bias:
                activation = np.append(activation, 1)

            preactivation = p @ activation

            activation = self.nl(preactivation)

        return activation



















