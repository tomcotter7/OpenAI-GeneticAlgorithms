import gym
import numpy as np
from agent import Agent

env = gym.make('HandManipulateBlockDense-v0').env
# env = gym.make('CartPole-v1')

n_hidden1 = 10
n_hidden2 = 20
n_input = 61

n_output = 1


# function to reshape the obs so that they can be passed into the network
def reshape_obs(obs, num=61):
    return np.reshape(obs, [1, num])


# function to create the initial generation
def create_initial_gen(n=10):
    gen = []
    for i in range(n):
        gen.append(Agent())
    return gen


# function to run a generation across an environment
def run_gen_env(env, gen):
    results = []
    for agent in gen:
        nn_specific_results = []
        for i in range(5):
            reward = run_env(env, agent.nn)
            nn_specific_results.append(reward)
        results.append(nn_specific_results)
    return gen, results


# function to run an agent in an environment
def run_env(env, nn):
    obs = env.reset()['observation']
    reward = -1
    done = False
    j = 0
    while not done and reward != 0:
        action = nn.predict(reshape_obs(obs))
        obs, reward, done, info = env.step(action[0])
        obs = obs['observation']
        if j == 75:
            break
        j += 1

    return reward


def get_best_nn(gen, gen_reward):
    # rank the neural networks based on the best reward?
    pass


initial_gen = create_initial_gen()
gen, gen_reward = run_gen_env(env, initial_gen)
print(gen, gen_reward)
