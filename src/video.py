import gym
import tensorflow as tf
import numpy as np
from stable_baselines3.common.atari_wrappers import *

ENV = gym.make("SpaceInvaders-v4", render_mode="human")
ENV = WarpFrame(ENV, 84)

nn = tf.keras.models.load_model("best_agent")

obs = ENV.reset()
done = False
while not done:
    action = nn.predict(np.array([obs]))
    action = np.argmax(action, axis=1)
    obs, reward, done, info = ENV.step(action[0])
