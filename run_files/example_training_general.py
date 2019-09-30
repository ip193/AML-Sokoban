import torch
from agent.DEEPSSRL import DEEPSSRL
from agent.SSRL import SSRL
import numpy as np
from agent.TrainingToolsGeneral import TrainingThreadGeneral, TrainingToolsGeneral

agents = [DEEPSSRL(layers=(2, 10, 10, 10, 1), nonlinearity=torch.tanh, start_at_1=False, use_special_binary_output=False,
                   use_abs_chi=True, use_abs_val_of_grad=False, use_argmax_out=False)]
        # SSRL(layers=(4, 1), nonlinearity=np.tanh, start_at_1=False, use_special_binary_output=True)]

#
# agents = [DEEPSSRL(layers=(4, 20, 20, 10, 1), nonlinearity=torch.tanh, use_special_binary_output=True, use_argmax_out=False),
#           DEEPSSRL(layers=(4, 20, 20, 10, 1), nonlinearity=torch.tanh, use_special_binary_output=True, use_abs_val_of_grad=True,
#                    use_argmax_out=False)]
          #SSRL(layers=(49, 4), nonlinearity=np.tanh)]

names = [# "torch_learner_max_diff",
         "deepMountainCarCont_LIKE_PAPER"]
        #"CartPole_deep_abs_everything"]

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

if len(agents) > 1:

    for agent in agents:
        training = TrainingThreadGeneral(agent, game_str)
        training.start()  # start the thread

else:
    while True:
        training = TrainingToolsGeneral(agents[0], game_str, num_training_episodes=int(5e4))
        training.runTraining()

