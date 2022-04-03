# Ideas

Ok, main idea:

- have a population which is a collection of random samples from the action space.
- run each population and collect the reward:
  - i.e. how far through the population does it get before failing / completion - we want the longest time before failing but the shortest time before completion ?? (maybe revise this - could be other factors we could look at)
  - other factors, how close to completion, highest reward received throughout process, etc...
- take the best populations - different choosing techniques (learn about these)
  - we have a population size of 25
  - we could take the top 5 (20%) to keep into the next population
  - then do a tournament selection to get the next 10 for crossover
  - then do a roulette wheel selection for mutation (10 Individuals)
  - that gives us a *new* 25 individuals to use in the next generation
- combine them (mutation, crossover, etc...)
- evolve the populations over time
