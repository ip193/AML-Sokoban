from agent.TrainingTools import TrainingTools
from agent.SSRL import SSRL
import pickle


agents = [SSRL()]  # , SSRL(layers=(100, 100, 50, 4), as_in_paper=False, special_update=True)]

for ind, agent in enumerate(agents):
    agent.setParams()  # initialize layer weights randomly
    agent.setSaveInfo(special_name_tag="example_learner")
    try:
        # agents[ind] = agent.load()  # if this is executed, an existing agent is loaded and trained if possible
        pass
    except Exception:
        pass

training = TrainingTools(agents, save_every=200)
database = "1567086188.896406"
training.setData(database)
training.setProtocol([1], [5e4])    # [1, 2, 3, 4], [2000, 2000, 2000, 2000])

training.runTraining(reload_every=400)

