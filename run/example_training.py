from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
import pickle


agents = [SSRL()]  # , SSRL(layers=(100, 100, 50, 4), as_in_paper=False, special_update=True)]

for ind, agent in enumerate(agents):
    agent.setParams()  # initialize layer weights randomly
    agent.setSaveInfo(special_name_tag="example_learner")
    try:
        agents[ind] = agent.load()
    except Exception:
        pass

training = TrainingTools(agents, save_every=50)
training.setData("example")
training.setProtocol([1], [5e3])    # [1, 2, 3, 4], [2000, 2000, 2000, 2000])

training.runTraining()

