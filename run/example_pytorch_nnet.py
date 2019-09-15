
import torch.nn.functional as F
import agent.DEEPSSRL
import numpy as np


agent = agent.DEEPSSRL.DEEPSSRL(layers=(100, 3, 4))
agent.setParams()

obs = np.random.rand(10, 10)

agent.giveObservation(obs)

f = agent.act()

print()