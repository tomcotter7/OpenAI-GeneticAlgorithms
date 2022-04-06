# import time
import random


class Individual():
    def __init__(self, env, num_actions, mr):
        self.env = env
        self.numActions = num_actions
        self.actions = []
        self.generate_actions()
        self.mr = mr
        self.fitness = 0

    def generate_actions(self):
        for a in range(self.numActions):
            self.actions.append(self.env.action_space.sample())

    # for now, this function will just return the average distance
    # between final acheived goal and desired goal
    def get_result(self):
        goal = self.env.reset()['desired_goal']
        for a in self.actions:
            self.env.render()
            obs, reward, done, info = self.env.step(a)
            if done:
                break
        if reward == 0:
            self.fitness = 0
        else:
            self.fitness = self.calculate_difference(goal, obs['achieved_goal'])
        return self.fitness

    def calculate_difference(self, goal, ach_goal):
        total_difference = 0
        for i in range(len(goal)):
            total_difference += abs(goal[i] - ach_goal[i])
        return total_difference / len(goal)

    def mutate(self):
        for index in range(len(self.actions)):
            prop = random.random()
            if (prop < self.mr):
                self.actions[index] = self.env.action_space.sample()
