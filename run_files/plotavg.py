import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from agent.DEEPSSRL import DEEPSSRL
from agent.SSRL import SSRL
import torch



load_agents = [DEEPSSRL(layers=(2, 10, 10, 10, 1), nonlinearity=torch.tanh, use_special_binary_output=True, use_argmax_out=False)]
          # DEEPSSRL(layers=(4, 20, 20, 10, 1), nonlinearity=torch.tanh, use_special_binary_output=True, use_abs_val_of_grad=True,
          #          use_argmax_out=False)]
          # #SSRL(layers=(49, 4), nonlinearity=np.tanh)]

names = [ #"deepSokoban_LIKE_PAPER"]
         "deepMountainCarCont_LIKE_PAPER"]
        #"CartPole_deep_abs_everything"]



def avg_by_segment(performance:list, segment_size:int = 100):
    performance = performance[:len(performance)-(len(performance)%segment_size)]
    performance = np.asarray(performance)
    performance = performance.reshape((-1, segment_size))
    avgs = np.apply_along_axis(np.mean, 1, performance)
    return avgs

for ind, l_a in enumerate(load_agents):
    l_a.setSaveInfo(special_name_tag=names[ind])
    load_agents[ind] = l_a.load()


def agent_plot_string(ind):
    return load_agents[ind].save_info.filename

step_distance = 1
segment_size=100
segment_size_test = 100


plot_mean_max = False
if plot_mean_max:

    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels

    # matplotlib.rcParams.update({'font.size': 15})

    plt.plot(load_agents[0].history.max_means)
    plt.title("Maximum means in network - " + load_agents[0].name )
    plt.ylabel(r"$max  |\mu_{b, ij}|$")
    plt.xlabel("Number of episodes trained")

    plt.show()

print()

# load_agents[0].save_info.filename = "DEEPSSRL_49_100_50_10_deepSokoban_LIKE_PAPER"

(numplots, numagents) = (1, len(load_agents))
fig, axes = plt.subplots(numagents, numplots)

def plot_all(X, y, label):
    if numplots == numagents == 1:
        a = axes
        a.plot(X[0], y[0], label=label)
        a.set_title(agent_plot_string(0))
        a.legend()

    elif numplots == 1 and numagents > 1:
        for a in range(numagents):
            ax = axes[a]
            ax.plot(X[a], y[a])
            ax.set_title(agent_plot_string(a))



y = [load_agents[load_agent_ind].history.training_rewards[step_distance] for load_agent_ind in range(len(load_agents))]
X = [list(range(len(rewards))) for rewards in y]

y = [avg_by_segment(rewards, segment_size) for rewards in y]
X = [avg_by_segment(episode_inds, segment_size) for episode_inds in X]

plot_all(X, y, "Training reward")


test_plot = False

if test_plot:

    y = [list(np.asarray(load_agents[load_agent_ind].history.testing_rewards[step_distance])[:, 1].T) for load_agent_ind in range(len(load_agents))]
    X = [list(np.asarray(load_agents[load_agent_ind].history.testing_rewards[step_distance])[:, 0].T) for load_agent_ind in range(len(load_agents))]

    y = [avg_by_segment(rewards, segment_size_test) for rewards in y]
    X = [avg_by_segment(episode_inds, segment_size_test) for episode_inds in X]

    plot_all(X, y, "Testing reward")


plt.xlabel("Episodes trained")
plt.ylabel("Average reward in " + str(segment_size) + " episodes.")
# plt.legend()
plt.show()

