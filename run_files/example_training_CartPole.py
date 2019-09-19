import torch
from agent.DEEPSSRL import DEEPSSRL
from agent.TrainingToolsGeneral import TrainingThreadGeneral

agents = [DEEPSSRL(layers=(4, 10, 10, 2), nonlinearity=torch.tanh),
               DEEPSSRL(layers=(4, 10, 10, 2), nonlinearity=torch.tanh, use_abs_update=False)]
names = ["CartPoleLearner", "CartPoleLearnerNoAbs"]

for ind, agent in enumerate(agents):

    agent.setParams()
    agent.setSaveInfo(special_name_tag=names[ind])

    try:
        agents[ind] = agent.load()  # if this is executed, an existing agent is loaded and trained if possible
        print("Loaded:", agents[ind].name)
        pass
    except Exception:
        print("Starting new agent:", agents[ind].name)
        pass

for agent in agents:
    training = TrainingThreadGeneral(agent)
    training.start()  # start the thread

