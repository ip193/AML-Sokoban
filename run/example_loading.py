from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import sklearn.kernel_ridge as krr

# Note: This is NOT the name of the data used. A single agent can learn from different datasets.
name = "example_learner"

load_agent = SSRL()
load_agent.setSaveInfo(special_name_tag=name)
load_agent = load_agent.load()

kernel = sklearn.metrics.pairwise.polynomial_kernel

reg = [LinearRegression(), krr.KernelRidge(kernel="rbf", alpha=0.3, gamma=300)]

y = np.asarray(load_agent.history.past_rewards).reshape(-1)
X = np.arange(y.size).reshape(-1, 1)

data_length = y.size

s = np.arange(0, data_length, 200)
y_, X_ = y[s], X[s]

print("Fitting plots...")
for ind, regr in enumerate(reg):
    print("(", ind, ")")
    regr.fit(X_, y_)
print("Done fitting plots.")

line_points = np.arange(0, data_length, 1000).reshape(-1, 1)

# Creates four polar axes, and accesses them through the returned array
numplots = len(reg)
fig, axes = plt.subplots(1, numplots)
for i in range(numplots):
    axes[i].plot(line_points, reg[i].predict(line_points))

plt.show()