from genetics.individual import Individual
import random


class Population:

    def __init__(self, pop_size, mutation_rate, crossover_rate, generations, env, prevPops):
        self.pop_size = pop_size
        self.mr = mutation_rate
        self.cr = crossover_rate
        self.gens = generations
        self.env = env
        self.pop = self.generate_pop()
        self.numberOfPrevPops = prevPops
        self.previousPops = []

    def generate_pop(self):
        pop = []
        for i in range(self.pop_size):
            ind = Individual(self.env, 5, self.mr)
            pop.append(ind)
        print("Initial Population Generated")
        return pop

    def get_topN(results, n):
        sorted_results = sorted(results, key=lambda tup: tup[1])
        return sorted_results[:n]

    def tournament_selection(self, results):
        for i in range(5):
            sample = random.sample(results, 10)
            best2 = self.get_topN(sample, 2)
            # do crossover here

    def roulette_wheel(results):
        # do roulette wheel selection

        # do mutation of selected indivs
        pass

    def run(self):
        for gen in self.gens:
            results = []
            for index, indivs in enumerate(self.pop):
                result = indivs.get_result()
                results.append((index, result))
            top5 = self.get_topN(results, 5)
