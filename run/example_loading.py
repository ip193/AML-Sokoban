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

#regr = LinearRegression()
regr = krr.KernelRidge(kernel="rbf", alpha=0.3, gamma=300)

y = np.asarray(load_agent.history.past_rewards).reshape(-1)
X = np.arange(y.size).reshape(-1, 1)

data_length = y.size

s = np.arange(3000, 6000)
y_, X_ = y[s], X[s]

regr.fit(X_, y_)

line_points = np.arange(0, data_length, 200).reshape(-1, 1)

plt.plot(line_points, regr.predict(line_points))
plt.scatter(line_points, y[line_points])
plt.show()