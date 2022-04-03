import gym
from genetics.pop import Population

POP_SIZE = 1
MUTATION_RATE = 0.4
CROSSOVER_RATE = 0.4
GENERATIONS = 50
PREV_POPS = 5

env_hmb = gym.make('HandManipulateBlock-v0')
env_hme = gym.make('HandManipulateEgg-v0')
env_hmp = gym.make('HandManipulatePen-v0')

p = Population(POP_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS, env_hmb, PREV_POPS)

print(p.run())
