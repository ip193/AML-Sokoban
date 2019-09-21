import numpy as np
import matplotlib.pyplot as plt
from agent.DEEPSSRL import DEEPSSRL
from agent.SSRL import SSRL
import torch

load_agents = [DEEPSSRL(layers=(2, 10, 10, 1), nonlinearity=torch.tanh, start_at_1=False, use_special_binary_output=False,
                   use_abs_chi=True, use_abs_val_of_grad=True, use_argmax_out=False)]
          # SSRL(layers=(4, 1), nonlinearity=np.tanh, start_at_1=False, use_special_binary_output=True)]

names = [# "torch_learner_max_diff",
         "deepMountainCar_abs_everything"]



def avg_by_segment(performance:list, segment_size:int = 100):
    performance = performance[:len(performance)-(len(performance)%segment_size)]
    performance = np.asarray(performance)
    performance = performance.reshape((-1, segment_size))
    avgs = np.apply_along_axis(np.mean, 1, performance)
    return avgs

for ind, l_a in enumerate(load_agents):
    l_a.setSaveInfo(special_name_tag=names[ind])
    load_agents[ind] = l_a.load()

step_distance = 1
segment_size=10


(numplots, numagents) = (1, len(load_agents))
fig, axes = plt.subplots(numagents, numplots)

def agent_plot_string(ind):
    return load_agents[ind].save_info.filename

def plot_all(X, y):
    if numplots == numagents == 1:
        a = axes
        a.plot(X[0], y[0])
        a.set_title(agent_plot_string(0))

    elif numplots == 1 and numagents > 1:
        for a in range(numagents):
            ax = axes[a]
            ax.plot(X[a], y[a])
            ax.set_title(agent_plot_string(a))


y = [load_agents[load_agent_ind].history.training_rewards[step_distance] for load_agent_ind in range(len(load_agents))]
X = [list(range(len(rewards))) for rewards in y]

y = [avg_by_segment(rewards, segment_size) for rewards in y]
X = [avg_by_segment(episode_inds, segment_size) for episode_inds in X]

plot_all(X, y)


# y = [list(np.asarray(load_agents[load_agent_ind].history.testing_rewards[step_distance])[:, 1].T) for load_agent_ind in range(len(load_agents))]
# X = [list(np.asarray(load_agents[load_agent_ind].history.testing_rewards[step_distance])[:, 0].T) for load_agent_ind in range(len(load_agents))]
#
# y = [avg_by_segment(rewards, segment_size) for rewards in y]
# X = [avg_by_segment(episode_inds, segment_size) for episode_inds in X]
#
# plot_all(X, y)

plt.show()

