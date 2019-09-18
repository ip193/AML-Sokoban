from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
from agent.DEEPSSRL import DEEPSSRL
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import sklearn.kernel_ridge as krr
from sklearn.gaussian_process import GaussianProcessRegressor

# Note: This is NOT the name of the data used. A single agent can learn from different datasets.
name = "torch_learner_3"


load_agents = [DEEPSSRL(layers=(49, 10, 10, 4), nonlinearity=torch.tanh)]
          #DEEPSSRL(layers=(49, 10, 10, 10, 4), nonlinearity=torch.tanh),
          #DEEPSSRL(layers=(49, 10, 10, 10, 10, 4), nonlinearity=torch.tanh)]

for ind, l_a in enumerate(load_agents):
    l_a.setSaveInfo(special_name_tag=name)
    load_agents[ind] = l_a.load()

# kernel = sklearn.metrics.pairwise.polynomial_kernel
kernel = sklearn.gaussian_process.kernels.RBF()
agent_regressors = [[LinearRegression(fit_intercept=False), LinearRegression(fit_intercept=False)]  # , krr.KernelRidge(kernel="rbf", alpha=0.3, gamma=300), GaussianProcessRegressor(kernel)]
                    for a in load_agents]

poly_n = 1
step_distance = 1

def build_poly_features(points, n):
    features = np.zeros((len(points), n+1)).astype(object)
    for i in range(n+1):
        features[:, i] = np.asarray([int(j**i) for j in points], dtype=object).reshape(-1)  # use python ints to handle overflow
    return features


def fit_agent_regression(load_agent_ind, step_distance, data_subset=100):

    y = np.asarray(load_agents[load_agent_ind].history.training_rewards[step_distance]).reshape(-1)
    X = build_poly_features(list(range(y.size)), poly_n)
    # FIXME technically inaccurate because there are testing episodes in between
    data_length = y.size

    s = np.arange(0, data_length, data_subset)  # select a subset of the data so fitting doesn't take so long
    y_, X_ = y[s], X[s]

    y_test = np.asarray(load_agents[load_agent_ind].history.testing_rewards[step_distance])[:, 1].reshape(-1)
    X_test = build_poly_features([int(j) for j in np.asarray(load_agents[load_agent_ind].history.testing_rewards[step_distance])[:, 0]], poly_n)
    data_length_test = y_test.size

    data_subset_test = int(data_subset//10)
    s_test = np.arange(0, data_length_test, data_subset_test)
    y__, X__ = y_test[s_test], X_test[s_test]

    print("Fitting plots...")
    for ind, regr in enumerate(agent_regressors[load_agent_ind]):  # each regression model
        print("(", ind, ")")
        if ind%2 == 0:
            regr.fit(X_, y_)
        else:
            regr.fit(X__, y__)
    print("Done fitting plots.")

    line_points_x = list(range(0, data_length, 100 ))  # evaluate regressors at these points
    return (line_points_x, build_poly_features(line_points_x, poly_n))


points = [fit_agent_regression(i, step_distance) for i in range(len(load_agents))]

numplots = (len(load_agents), len(agent_regressors[0])//2)
fig, axes = plt.subplots(*numplots)

for r in range(2):
    axes.plot(points[0][0], agent_regressors[0][r].predict(points[0][1]))

plt.show()
#
# for a in range(numplots[0]):
#     for r in range(numplots[1]):
#         for i in range(2):
#             if numplots[0] == 1:
#                 if numplots[1] == 1:
#                     axes.plot(points[a][0], agent_regressors[a][2*r + i].predict(points[a][1]))
#                 else:
#                     axes[r].plot(points[a][0], agent_regressors[a][2*r + i].predict(points[a][1]))
#             else:
#                 if numplots[1] == 1:
#                     axes[a].plot(points[a][0], agent_regressors[a][2*r + i].predict(points[a][1]))
#                 else:
#                     axes[a, r].plot(points[a][0], agent_regressors[a][2*r + i].predict(points[a][1]))
#
# plt.show()