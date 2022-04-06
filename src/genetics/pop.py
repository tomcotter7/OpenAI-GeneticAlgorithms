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
        self.nextPop = []

    def generate_pop(self):
        pop = []
        for i in range(self.pop_size):
            ind = Individual(self.env, 5, self.mr)
            pop.append(ind)
        print("Initial Population Generated")
        return pop

    # gets the best performing individual from the population based on its
    # L2 distance from the "goal" - abs(goal - acheived-goal)
    def get_topN(results, n):
        sorted_results = sorted(results, key=lambda tup: tup[1])
        return sorted_results[:n]

    # uniform crossover function
    def crossover(self, p1, p2):
        # we need to assign a weight to the slightly better parent.
        # then randomly take each one
        # (or we could do 0.6 / 0.4 in favour of the better parent)
        # let's first try with no weighting and see what happens.
        children = [Individual(self.env, 0, self.mr), Individual(self.env, 0, self.mr)]
        if p1.fitness >= p2.fitness:
            best_parent = p1
            worst_parent = p2
        else:
            best_parent = p2
            worst_parent = p1
        for i in range(2):
            for a in range(len(p1.actions)):
                if random.uniform() >= 0.6:
                    children[i].actions.append(best_parent.actions[a])
                else:
                    children[i].actions.append(worst_parent.actions[a])
        return children[0], children[1]

    def tournament_selection(self, results):
        children = []
        for i in range(5):
            sample = random.sample(results, 10)
            best2 = self.get_topN(sample, 2)
            c1, c2 = self.crossover(best2)
            children.append(c1)
            children.append(c2)
        return children

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
            crossover_children = self.tournament_selection(results)
