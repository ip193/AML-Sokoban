from agent.TrainingTools import TrainingThread
from agent.SSRL import SSRL
from agent.DEEPSSRL import DEEPSSRL
import pickle
import numpy as np
import torch


agents = [DEEPSSRL(layers=(100, 10, 10, 4), nonlinearity=torch.tanh)]

for ind, agent in enumerate(agents):
    agent.setParams()  # initialize layer weights randomly
    agent.setSaveInfo(special_name_tag="torch_learner")
    try:
        agents[ind] = agent.load()  # if this is executed, an existing agent is loaded and trained if possible
        print("Loaded:", agents[ind].name)
        pass
    except Exception:
        print("Starting new agent:", agents[ind].name)
        pass

for agent in agents:

    training = TrainingThread([agent], save_every=200, reload_every=400)
    database = "main"
    training.training_tools.setData(database)
    training.training_tools.setProtocol([1, 2, 3, 4], [1e5, 1e5, 1e5, 1e5])

    training.start()  # start the thread

