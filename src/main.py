import gym
import numpy as np
import matplotlib.pyplot as plt
from runner import Runner


ENV = gym.make("SpaceInvaders-v4")

sp_runner = Runner(ENV, 2, experiences_per_gen=2, pop_size=2)
gen, gen_reward, best_scores, mean_scores, median_scores = sp_runner.run_genetic_algorithm()

plt.figure(figsize=(10, 10))
plt.plot(best_scores, label="Best Score")
plt.plot(mean_scores, label="Mean Score")
plt.plot(median_scores, label="Median Score")
plt.legend(loc="upper right")

NEW_ENV = gym.make("DemonAttack-v4", render_mode="human")

final_best_agent_name = max(gen_reward, key=lambda x: x[1])[0]
best_agent = [agent for agent in gen if agent.name == final_best_agent_name][0]
da_runner = Runner(NEW_ENV, 10, experiences_per_gen=2, pop_size=2)

random_results = da_runner.run_multiple_times(random=True)
model_results = da_runner.run_multiple_times(model=best_agent.nn)

print("Random Results: Mean: {}, Best: {}".format(
    np.mean(random_results), max(random_results)))
print("Model Results: Mean: {}, Best: {}".format(
    np.mean(model_results), max(model_results)))
