import gym
import numpy as np
from agent import Agent
import random

env = gym.make('HandManipulateBlockDense-v0').env
# env = gym.make('CartPole-v1')

GENERATIONS = 10
POP_SIZE = 2


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


# how do we do this?
def crossover(agent1, agent2, cr=0.3):
    pass


def mutate_layer_weight(lw, mr):
    chance = random.uniform(0, 1)
    if chance < mr:
        lw[0] = np.vectorize(lambda x: np.multiply(x, random.uniform(0.5, 1.5)))(lw[0])
    return lw


def mutate(agent, mr=0.5):
    # mutate the agent by multiplying a random set of weights by a
    # random number between 0.5 and 1.5.
    current_weights = agent.get_weights()
    new_weights = [mutate_layer_weight(lw, mr) for lw in current_weights]
    agent.update_weights(new_weights)
    return agent


def get_best_nn(gen, gen_reward):
    # rank the neural networks based on the best reward?
    # lets do a tournament selection basd on the average reward

    # median_values = np.median(gen_reward, axis=1)
    mean_values = sorted(np.mean(gen_reward, axis=1))
    new_gen = []
    for i in range(int(len(gen) / 2)):
        tournament_1 = random.sample(mean_values, int(len(mean_values) / 2))
        tournament_2 = random.sample(mean_values, int(len(mean_values) / 2))
        p1 = gen[mean_values.index(max(tournament_1))]
        p2 = gen[mean_values.index(max(tournament_2))]
        p1 = mutate(p1)
        p2 = mutate(p2)
        new_gen.append(p1)
        new_gen.append(p2)

    return new_gen


initial_gen = create_initial_gen(n=POP_SIZE)
gen, gen_reward = run_gen_env(env, initial_gen)
for g in range(GENERATIONS):
    gen = get_best_nn(gen, gen_reward)
    gen, gen_reward = run_gen_env(env, gen)
