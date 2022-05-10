import random
import numpy as np
from stable_baselines3.common.atari_wrappers import *
import math
from agent import Agent


class Runner:

    def __init__(self, env, generations, experiences_per_gen=8, pop_size=10,
                 initial_gen=None, no_limit=False):
        self.env = WarpFrame(env, 84)
        if initial_gen is None:
            self.gen = self.create_initial_gen(pop_size)
        else:
            self.gen = initial_gen
        self.generations = generations
        self.experience = experiences_per_gen
        self.nolimit = no_limit

    # function to create an initial generation
    def create_initial_gen(self, n):
        return [Agent("agent"+str(i)) for i in range(n)]

    # function to run a generation across an environment
    def run_gen_env(self, gen):
        results = []
        for agent in gen:
            nn_specific_results = [self.run_env(self.env, agent.nn)
                                   for i in range(self.experience)]
            print("Agent {} with scores {}".format(agent.name, nn_specific_results))
            results.append((agent.name, 0.3 * np.median(nn_specific_results,
                           axis=0) + 0.7 * np.mean(nn_specific_results, axis=0)))
        print("Results {}".format(results))
        return gen, results

    # function to run an agent in an environment
    def run_env(self, env, nn):
        obs = env.reset()
        reward = -1
        award = 0
        done = False
        while not done:
            action = nn.predict(np.array([obs]))
            action = np.argmax(action, axis=1)
            obs, reward, done, info = env.step(action[0])
            if info['episode_frame_number'] >= 15000:
                break
            if not self.nolimit and info['lives'] < 3:
                break
            award += reward

        return award

    # crossover functions
    def crossover_lwb(self, wbs1, wbs2, cr):
        for j in range(len(wbs1)):
            chance = random.uniform(0, 1)
            if chance < cr:
                temp = wbs1[j].copy()
                wbs1[j] = wbs2[j]
                wbs2[j] = temp

        return wbs1, wbs2

    def crossover(self, weights1, weights2, cr):
        new_weights1 = []
        new_weights2 = []
        for i in range(len(weights1)):
            if weights1[i] != []:
                lw1, lw2 = self.crossover_lwb(weights1[i][0], weights2[i][0], cr)
                lb1, lb2 = self.crossover_lwb(weights1[i][1], weights2[i][1], cr)
                new_weights1.append([lw1, lb1])
                new_weights2.append([lw2, lb2])
            else:
                new_weights1.append([])
                new_weights2.append([])
        return new_weights1, new_weights2

    def mutate_wb(self, lw, mr):
        chance = random.uniform(0, 1)
        if chance < mr:
            # change by a percentage
            lw = np.vectorize(lambda x: np.multiply(x, random.uniform(0.5, 1.5)))(lw)

            # change the sign of the weights
            # lw = np.vectorize(lambda x: np.multiply(x, -1))(lw)

            # replace with a random value ?
            # lw = np.random.random(lw.shape)

        return lw

    def mutate_lw(self, lw, mr):
        return [self.mutate_wb(wb, mr) for wb in lw]

    def mutate(self, weights, mr):
        return [self.mutate_lw(lw, mr) for lw in weights]

    def just_mutation(self, agent, mr):
        new_weights = self.mutate(agent.get_weights(), mr)
        child = Agent("child")
        child.update_weights(new_weights)
        return child

    def run_evolution(self, agent1, agent2, cr, mr):
        cweights1, cweights2 = self.crossover(
            agent1.get_weights(), agent2.get_weights(), cr)

        mweights1 = self.mutate(cweights1, mr)
        mweights2 = self.mutate(cweights2, mr)

        c1 = Agent("child")
        c2 = Agent("child")

        c1.update_weights(mweights1)
        c2.update_weights(mweights2)

        return c1, c2

    def get_best_n_names(self, lst, n=2):
        best_n_names = [name for name, _, in sorted(lst, key=lambda x: x[1])[-n:]]
        return best_n_names

    def get_new_gen(self, gen_reward, g):

        new_gen = []

        mr = 0.3 / math.sqrt(g + 1)
        cr = 0.35

        print("Current MR: {} and Current CR: {}".format(mr, cr))

        # initially let's perform elitism and select the best 2 individuals.
        best_2_names = self.get_best_n_names(gen_reward)
        print("2 best agents from generation:", best_2_names)
        best_agents = [self.just_mutation(indiv, mr=mr)
                       for indiv in self.gen if indiv.name in best_2_names]
        new_gen.append(best_agents[0])
        new_gen.append(best_agents[1])

        # now let's perform tournament selection n times

        for i in range(int(len(self.gen) / 2) - 1):
            tournament_1 = random.sample(gen_reward, int(len(gen_reward) / 2))
            # take the two best parents from this tournament.
            best_2 = self.get_best_n_names(tournament_1)
            agents = [indiv for indiv in self.gen if indiv.name in best_2]
            # perform crossover
            c1, c2 = self.run_evolution(agents[0], agents[1], cr, mr)
            new_gen.append(c1)
            new_gen.append(c2)

        for index, agent in enumerate(new_gen):
            new_name = "agent{}".format(str(index))
            agent.update_name(new_name)

        self.gen = new_gen

    def best_score(self, gen_reward):
        best_score = [score for _, score in sorted(gen_reward, key=lambda x: x[1])][-1]
        return best_score

    def run_genetic_algorithm(self):
        best_scores = []
        mean_scores = []
        median_scores = []
        self.gen, gen_reward = self.run_gen_env(self.gen)
        for g in range(self.generations):
            print("Generation %i" % (g))
            best_scores.append(self.best_score(gen_reward))
            mean_scores.append(np.mean([score for _, score in gen_reward]))
            median_scores.append(np.median([score for _, score in gen_reward]))
            print("Scores:", gen_reward)
            self.get_new_gen(gen_reward, g)
            self.gen, gen_reward = self.run_gen_env(self.gen)

        return self.gen, gen_reward, best_scores, mean_scores, median_scores

    def run_multiple_times(self, model):
        return [self.run_env(self.env, model) for i in range(self.experience)]

    def run_random_algorithm(self, env):
        obs = env.reset()
        award = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if info['episode_frame_number'] >= 15000:
                break
            award += reward

        return award
