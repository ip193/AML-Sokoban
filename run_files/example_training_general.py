import torch
from agent.DEEPSSRL import DEEPSSRL
from agent.SSRL import SSRL
import numpy as np
from agent.TrainingToolsGeneral import TrainingThreadGeneral

agents = [DEEPSSRL(layers=(2, 10, 10, 1), nonlinearity=torch.tanh, start_at_1=False, use_special_binary_output=False,
                   use_abs_chi=True, use_abs_val_of_grad=True, use_argmax_out=False)]
          # SSRL(layers=(4, 1), nonlinearity=np.tanh, start_at_1=False, use_special_binary_output=True)]

names = [# "torch_learner_max_diff",
         "deepMountainCar_abs_everything"]

game_str = "MountainCarContinuous-v0"  # "CartPole-v0"

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
    training = TrainingThreadGeneral(agent, game_str)
    training.start()  # start the thread

