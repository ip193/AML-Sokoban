from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
import matplotlib.pyplot as plt

load_agent = SSRL()
load_agent.setSaveInfo(special_name_tag="example_learner")
load_agent = load_agent.load()

plt.plot(load_agent.history.past_rewards)