from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import sklearn.kernel_ridge as krr
from sklearn.gaussian_process import GaussianProcessRegressor

# Note: This is NOT the name of the data used. A single agent can learn from different datasets.
name = "example_learner"

load_agent = SSRL()
load_agent.setSaveInfo(special_name_tag=name)
load_agent = load_agent.load()

# kernel = sklearn.metrics.pairwise.polynomial_kernel
kernel = sklearn.gaussian_process.kernels.RBF()

reg = [LinearRegression(fit_intercept=False), krr.KernelRidge(kernel="rbf", alpha=0.3, gamma=300), GaussianProcessRegressor(kernel)]

def build_poly_features(size, n):
    features = np.zeros((size, n+1)).astype(object)
    for i in range(n+1):
        features[:, i] = np.asarray([int(j**i) for j in range(size)], dtype=object).reshape(-1)  # use python ints to handle overflow
    return features

poly_n = 8
y = np.asarray(load_agent.history.past_rewards).reshape(-1)
X = build_poly_features(y.size, poly_n)


data_length = y.size

s = np.arange(0, data_length, 100)  # select a subset of the data so fitting doesn't take so long
y_, X_ = y[s], X[s]

print("Fitting plots...")
for ind, regr in enumerate(reg):
    print("(", ind, ")")
    regr.fit(X_, y_)
print("Done fitting plots.")

line_points_x = np.arange(0, data_length, 500)  # evaluate regressors at these points
line_points = build_poly_features(data_length, poly_n)[line_points_x]

numplots = len(reg)
fig, axes = plt.subplots(1, numplots)
for i in range(numplots):
    axes[i].plot(line_points_x, reg[i].predict(line_points))

plt.show()