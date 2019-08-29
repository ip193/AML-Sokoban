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

kernel = sklearn.metrics.pairwise.rbf_kernel

# regr = LinearRegression()
regr = krr.KernelRidge(kernel=kernel, gamma=1)

y = np.asarray(load_agent.history.past_rewards).reshape(-1, 1)
X = np.arange(y.size).reshape(-1, 1)

regr.fit(X, y)

line_points = np.arange(0, y.size, 100).reshape(-1, 1)

plt.plot(line_points, regr.predict(line_points))
plt.scatter(line_points, y[line_points])
plt.show()