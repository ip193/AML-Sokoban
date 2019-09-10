from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
import pickle
import numpy as np


agents = [SSRL(), SSRL(layers=(100, 100, 50, 4), as_in_paper=False, special_update=True)]

for ind, agent in enumerate(agents):
    agent.setParams()  # initialize layer weights randomly
    agent.setSaveInfo(special_name_tag="compare_learners")
    try:
        agents[ind] = agent.load()  # if this is executed, an existing agent is loaded and trained if possible
        print("Loaded:", agents[ind].name)
        pass
    except Exception:
        print("Starting new agent:", agents[ind].name)
        pass

training = TrainingTools(agents, save_every=200)
database = "main"
training.setData(database)
training.setProtocol([1, 2], [2e4, 2e4])    # [1, 2, 3, 4], [2000, 2000, 2000, 2000])

training.runTraining(reload_every=400)
#  training.runTraining()

