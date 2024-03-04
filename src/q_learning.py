import gymnasium as gym
import numpy as np
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

env = gym.make("FrozenLake-v1", is_slippery=True)

learning_rate = 0.85
discount_factor = 0.99

Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000

rList = []
for i in range(num_episodes):
    state, info = env.reset()
    rAll = 0
    terminated = False
    
    while not terminated:
        action = np.argmax(Q[state, :] + np.random.rand(1, env.action_space.n) / (i + 1))
        
        observation, reward, terminated, truncated, info = env.step(action)

        if truncated:
            break
        
        Q[state, action] = (1 - learning_rate) * Q[state, action] \
                            + learning_rate * (reward + discount_factor * np.max(Q[observation,:]))
        
        rAll += reward
        state = observation
        
    if terminated:
        rList.append(rAll)
env.close()

# print(Q)
print(f'All Result: {sum(rList)/num_episodes}')
print(f'Last 10% : {sum(rList[-int(len(rList)*0.1):])/(int(len(rList)*0.1))}')