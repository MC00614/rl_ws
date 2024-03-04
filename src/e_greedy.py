import gymnasium as gym
import numpy as np
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

env = gym.make("FrozenLake-v1", is_slippery=False)

discount_factor = 0.99

Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000

rList = []
for i in range(num_episodes):
    state, info = env.reset()
    rAll = 0
    terminated = False
    
    e = 1.0 / ((i // 100) + 1)
    
    while not terminated:
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = rargmax(Q[state, :])
        
        observation, reward, terminated, truncated, info = env.step(action)

        if truncated:
            break
        
        Q[state, action] = reward + discount_factor * np.max(Q[observation,:])
        
        rAll += reward
        state = observation
        
    if terminated:
        rList.append(rAll)
env.close()

print(f'All : {sum(rList)/num_episodes}')
print(f'Last Half : {sum(rList[-len(rList)//2:])/(len(rList)/2)}')