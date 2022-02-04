from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
seed = 42
model = keras.models.load_model('./models/cnn.h5')
env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)
env = gym.wrappers.Monitor(env, './videos/', video_callable = lambda episode_id: True , force=True )
n_episodes = 10
returns = []
for _ in range (n_episodes) :
  ret = 0
  state = np.array(env.reset())
  done = False
  while not done :
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_probs = model(state_tensor, training=False)
    action = tf.argmax(action_probs[0]).numpy()
    state_next, reward, done, _ = env.step(action)
    state_next = np.array(state_next)
    ret += reward
    state = state_next
  returns.append(ret)
env.close()
print('Returns : {}'.format(returns))