from agent.TrainingToolsSokoban import TrainingThreadSokoban
from agent.SSRL import SSRL
from agent.DEEPSSRL import DEEPSSRL
import pickle
import numpy as np
import torch


agents = [#DEEPSSRL(layers=(49, 10, 10, 4), nonlinearity=torch.tanh),
          DEEPSSRL(layers=(49, 10, 10, 4), nonlinearity=torch.tanh)]

names = [# "torch_learner_max_diff",
         "master"]
for ind, agent in enumerate(agents):
    agent.setParams()  # initialize layer weights randomly
    agent.setSaveInfo(special_name_tag=names[ind])
    try:
        agents[ind] = agent.load()  # if this is executed, an existing agent is loaded and trained if possible
        print("Loaded:", agents[ind].name)
        pass
    except Exception:
        print("Starting new agent:", agents[ind].name)
        pass

for agent in agents:

    training = TrainingThreadSokoban([agent], save_every=200, reload_every=None, test_every=10)
    database = "changed_generate_env_main7x7-2"
    training.training_tools.setData(database)
    training.training_tools.loadData()
    training.training_tools.loadData(test=True)
    training.training_tools.setProtocol([1, 2, 3, 4], [1e5, 1e5, 1e5, 1e5])

    training.start()  # start the thread

