import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FFNN(nn.Module):
    """
    torch-based NNET for backpropagation
    """

    def __init__(self):
        super(FFNN, self).__init__()
