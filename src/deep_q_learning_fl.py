import gymnasium as gym
import numpy as np
import tensorflow as tf

env = gym.make("FrozenLake-v1", is_slippery=True)

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1
discount_factor = 0.99
num_episodes = 5000

class QNetwork(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.W = tf.Variable(tf.random.uniform([input_size, output_size]), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.W)

X = tf.keras.Input(shape=[1,input_size], dtype=tf.float32)
Y = tf.keras.Input(shape=[1, output_size], dtype=tf.float32)

model = QNetwork(input_size, output_size)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)


rList = []

for i in range(num_episodes):
    state, info = env.reset()
    state = np.eye(input_size)[state:state+1]
    # state = np.reshape(state, [1, input_size])
    e = 1. / ((i / 50) + 10)
    rAll = 0
    terminated = False
    
    while not terminated:
        with tf.GradientTape() as tape:
            Qs = model(state.astype('float32'))
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)
            
            observation, reward, terminated, truncated, info = env.step(action)
            observation = np.eye(input_size)[observation:observation+1]
            # observation = np.reshape(observation, [1, input_size])
            
            Qo = Qs.numpy()
            if terminated:
                Qo[0, action] = reward
            else:
                Qs1 = model(observation.astype('float32'))
                Qo[0, action] = reward + discount_factor * np.max(Qs1.numpy())
                
            loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(Qo, dtype=tf.float32) - Qs))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            rAll += reward
            state = observation
            
        
    if terminated:
        rList.append(rAll)
env.close()

# print(Q)
print(f'All Result: {sum(rList)/num_episodes}')
print(f'Last 10% : {sum(rList[-int(len(rList)*0.1):])/(int(len(rList)*0.1))}')