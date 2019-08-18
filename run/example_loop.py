
import gym
import gym_sokoban
import time


env_name = 'Sokoban-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))



for i_episode in range(1):#20
    observation = env.reset()

    for t in range(100):#100
        env.render(mode='human')
        a = observation
        b = env.room_fixed
        c = env.room_state
        action = env.action_space.sample()

        # Sleep makes the actions visible for users
        time.sleep(1)
        observation, reward, done, info = env.step(action)

        print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break

    env.close()

time.sleep(10)