


import numpy as np


class FFNN:
    """
    Implementation of a simple feed-forward neural net
    """

    def __init__(self, nonlinearity, bias=True):

        """
        :param nonlinearity: function handle for activation function
        :param bias: True if a bias node should be added to each layer
        """

        self.weights = None
        self.nl = nonlinearity
        self.bias = bias

    def setWeights(self, weights):
        self.weights = weights

    def forward(self, input):
        """
        Return the activations of each layer in the network (including input) in a list
        :param input:
        :return:
        """

        input = input.flatten()
        if input.shape[0] + int(self.bias) != self.weights[0].shape[1]:
            raise RuntimeError("Incorrect input dimensions. Got: " + input.shape + " Expected: " + self.weights[0].shape)

        all_activations = []

        activation = input  # holds the activation of the previous layer

        for ind, w in enumerate(self.weights):

            if self.bias:
                activation = np.append(activation, 1.)

            all_activations.append(activation)

            preactivation = w @ activation

            activation = self.nl(preactivation)

        all_activations.append(activation)

        return all_activations



















