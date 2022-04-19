from agent import Agent
import numpy as np
import random


def mutate_layer_weight(lw, mr):
    chance = random.uniform(0, 1)
    print("Old Layer Weight:", lw)
    if chance < mr:
        lw[0] = np.vectorize(lambda x: np.multiply(x, random.uniform(0.5, 1.5)))(lw[0])
    print("New Layer Weight:", lw)
    return lw


agent = Agent()
mr = 1.0
weights = agent.get_weights()

new_weights = [mutate_layer_weight(lw, mr) for lw in weights]
agent.update_weights(new_weights)
