import threading
import gym
import torch
import numpy as np

class TrainingToolsGeneral():
    """
    Training class for the game Lunar Lander
    """

    def __init__(self, agent, steps=100, save_every=100, num_training_episodes=int(1e4)):

        self.agent = agent  # Note: only implemented for one agent

        self.steps = steps
        self.save_every = save_every
        self.num_training_episodes = num_training_episodes

    def runTraining(self):
        """
        Execute the training process
        :return:
        """
        env = gym.make("CartPole-v0")
        agent = self.agent

        for episodes in range(self.num_training_episodes):

            agent.resetEpisode()
            observation = env.reset()

            for t in range(self.steps):
                # env.render()
                agent.giveObservation(observation)
                action = agent.act()
                observation, reward, done, info = env.step(action)
                agent.giveReward(reward)
                agent.incrementTimeStep()

                if done:
                    break

            agent.endOfEpisodeUpdate(1)

            if episodes % self.save_every == 0:
                print("Saving agent")
                print("Total episodes for agent:", len(agent.history.training_rewards[1]))
                agent.save()


class TrainingThreadGeneral(threading.Thread):
    """
    Runs training in its own thread (should pass object)
    """
    def __init__(self, agent, **kwargs):
        super().__init__()
        self.training_tools = TrainingToolsGeneral(agent, **kwargs)

    def run(self):
        print("Thread starting:", self.training_tools.agent.save_info.filename)
        self.training_tools.runTraining()








