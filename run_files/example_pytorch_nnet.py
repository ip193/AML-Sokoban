import torch
import torch.nn.functional as F
import agent.DEEPSSRL
import numpy as np

def identity(x):
    return x

agent = agent.DEEPSSRL.DEEPSSRL(layers=(100, 1, 2), nonlinearity=identity)
agent.setParams()

obs = np.random.rand(10, 10)

agent.resetEpisode()
agent.giveObservation(obs)

f = agent.act()

print()