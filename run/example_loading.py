from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

load_agent = SSRL()
load_agent.setSaveInfo(special_name_tag="example_learner")
load_agent = load_agent.load()

regr = LinearRegression()
y = load_agent.history.past_rewards
y = np.asarray(y).reshape(-1, 1)

regr.fit(np.arange(y.size).reshape(-1, 1), y)

line_points = np.arange(0, y.size, 100).reshape(-1, 1)

plt.plot(line_points, regr.predict(line_points))
plt.scatter(line_points, y[line_points])
plt.show()