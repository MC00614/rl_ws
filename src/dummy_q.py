import gymnasium as gym
import numpy as np
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

env = gym.make("FrozenLake-v1", is_slippery=False)

Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000

rList = []
for _ in range(num_episodes):
    state, info = env.reset()
    rAll = 0
    terminated = False
    while not terminated:
        action = rargmax(Q[state, :])
        observation, reward, terminated, truncated, info = env.step(action)

        if truncated:
            break
        
        Q[state, action] = reward + np.max(Q[observation,:])
        
        rAll += reward
        state = observation
        
    if terminated:
        rList.append(rAll)
env.close()

print(rList)
print(Q)
print(sum(rList)/num_episodes)