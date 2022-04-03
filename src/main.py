import gym


def run_env(env):
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timestamps".format(t+1))
                break
    env.close()


def create_populations(env, sizeOfPop):
    pop = []
    for i in range(sizeOfPop):
        pop.append(env.action_space.sample())
    return pop


env = gym.make('HandManipulateBlock-v0')
ps = create_populations(env, 20)
# run_env(env)
