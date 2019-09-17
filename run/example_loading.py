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
name = "torch_learner_2"


load_agents = [DEEPSSRL(layers=(49, 10, 10, 4), nonlinearity=torch.tanh),
          DEEPSSRL(layers=(49, 10, 10, 10, 4), nonlinearity=torch.tanh),
          DEEPSSRL(layers=(49, 10, 10, 10, 10, 4), nonlinearity=torch.tanh)]

for ind, l_a in enumerate(load_agents):
    l_a.setSaveInfo(special_name_tag=name)
    load_agents[ind] = l_a.load()

# kernel = sklearn.metrics.pairwise.polynomial_kernel
kernel = sklearn.gaussian_process.kernels.RBF()
agent_regressors = [[LinearRegression(fit_intercept=False)]  # , krr.KernelRidge(kernel="rbf", alpha=0.3, gamma=300), GaussianProcessRegressor(kernel)]
                    for a in load_agents]

poly_n = 4
step_distance = 1

def build_poly_features(size, n):
    features = np.zeros((size, n+1)).astype(object)
    for i in range(n+1):
        features[:, i] = np.asarray([int(j**i) for j in range(size)], dtype=object).reshape(-1)  # use python ints to handle overflow
    return features


def fit_agent_regression(load_agent_ind, step_distance):

    y = np.asarray(load_agents[load_agent_ind].history.past_rewards[step_distance]).reshape(-1)
    X = build_poly_features(y.size, poly_n)
    data_length = y.size

    s = np.arange(0, data_length, 1)  # select a subset of the data so fitting doesn't take so long
    y_, X_ = y[s], X[s]

    print("Fitting plots...")
    for ind, regr in enumerate(agent_regressors[load_agent_ind]):  # each regression model
        print("(", ind, ")")
        regr.fit(X_, y_)
    print("Done fitting plots.")

    line_points_x = np.arange(0, data_length, 5)  # evaluate regressors at these points
    return (line_points_x, build_poly_features(data_length, poly_n)[line_points_x])


points = [fit_agent_regression(i, step_distance) for i in range(len(load_agents))]

numplots = (len(load_agents), len(agent_regressors[0]))
fig, axes = plt.subplots(*numplots)

for a in range(numplots[0]):
    for r in range(numplots[1]):
        if numplots[0] == 1:
            if numplots[1] == 1:
                axes.plot(points[a][0], agent_regressors[a][r].predict(points[a][1]))
            else:
                axes[r].plot(points[a][0], agent_regressors[a][r].predict(points[a][1]))
        else:
            if numplots[1] == 1:
                axes[a].plot(points[a][0], agent_regressors[a][r].predict(points[a][1]))
            else:
                axes[a, r].plot(points[a][0], agent_regressors[a][r].predict(points[a][1]))

plt.show()