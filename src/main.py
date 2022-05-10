import gym
import numpy as np
import matplotlib.pyplot as plt
from runner import Runner
from agent import Agent

ENV = gym.make("SpaceInvaders-v4")

sp_runner = Runner(ENV, generations=25, experiences_per_gen=10, pop_size=12)
gen, gen_reward, best_scores, mean_scores, median_scores = sp_runner.run_genetic_algorithm()


plt.figure(figsize=(10, 10))
plt.plot(best_scores, label="Best Score")
plt.plot(mean_scores, label="Mean Score")
plt.plot(median_scores, label="Median Score")
plt.legend(loc="upper right")

final_best_agent_name = max(gen_reward, key=lambda x: x[1])[0]
best_agent = [agent for agent in gen if agent.name == final_best_agent_name][0]

best_agent.nn.save("/content/drive/MyDrive/year3/dia/best_agent")
best_agent.nn.save_weights("/content/drive/MyDrive/year3/dia/best_agent_ckpt")

# best_nn = tf.keras.models.load_model("/content/drive/MyDrive/year3/dia/best_agent")
best_nn = best_agent.nn

C_ENV = gym.make("Carnival-v4")
SP_ENV = gym.make("SpaceInvaders-v4")
DA_ENV = gym.make("DemonAttack-v4")

sp_runner = Runner(SP_ENV, generations=1, experiences_per_gen=10,
                   pop_size=1, no_limit=True)
da_runner = Runner(DA_ENV, generations=1, experiences_per_gen=10,
                   pop_size=1, no_limit=True)
c_runner = Runner(C_ENV,
                  generations=1, experiences_per_gen=10, pop_size=1, no_limit=True)

random_agent = Agent("random")
random_nn = random_agent.nn

random_results = sp_runner.run_multiple_times(model=random_nn)
model_results = sp_runner.run_multiple_times(model=best_nn)

print("Space Invaders:")

print("Random Results: Mean: {}, Median: {}, Best: {}".format(
    np.mean(random_results), np.median(random_results), max(random_results)))
print("Model Results: Mean: {}, Median: {}, Best: {}".format(
    np.mean(model_results), np.median(model_results), max(model_results)))

print("-----------------")

print("Demon Attack")

random_results = da_runner.run_multiple_times(model=random_nn)
model_results = da_runner.run_multiple_times(model=best_nn)

print("Random Results: Mean: {}, Median: {}, Best: {}".format(
    np.mean(random_results), np.median(random_results), max(random_results)))
print("Model Results: Mean: {}, Median: {}, Best: {}".format(
    np.mean(model_results), np.median(model_results), max(model_results)))

print("--------------")

print("Carnival")

random_results = c_runner.run_multiple_times(model=random_nn)
model_results = c_runner.run_multiple_times(model=best_nn)

print("Random Results: Mean: {}, Median: {}, Best: {}".format(
    np.mean(random_results), np.median(random_results), max(random_results)))
print("Model Results: Mean: {}, Median: {}, Best: {}".format(
    np.mean(model_results), np.median(model_results), max(model_results)))
