import gym
from genetics.pop import Population

# this is the max number of actions the environment can do before done is set to true
POP_SIZE = 100
MUTATION_RATE = 0.4
CROSSOVER_RATE = 0.4
GENERATIONS = 50
PREV_POPS = 5

env_hmb = gym.make('HandManipulateBlock-v0')
env_hme = gym.make('HandManipulateEgg-v0')
env_hmp = gym.make('HandManipulatePen-v0')

# p = Poulation(POP_SIZE, MUTATION_RATE, CROSSOVER_RATE, GENERATIONS, env_hmb, PREV_POPS)


for episode in range(20):
    obs = env_hmb.reset()
    for t in range(200):
        if t % 50 == 0:
            print(t)
        env_hmb.render()
        action = env_hmb.action_space.sample()
        obs, reward, done, info = env_hmb.step(action)
        if done:
            print("Episode finshed after {} timestamps", format(t+1))
            print("Reward:", reward)
            break
