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
    - how to do crossover? take half from each, or something more clever
      - let's try uniform crossover first - could compare this to the other crossovers if i have time (one point crossover, multiple point crossover).
      - then Davis' Order Crossover (OX1) - this might not work just because of the order the actions have to be taken?
  - then do a roulette wheel selection for mutation (10 Individuals)
  - that gives us a *new* 25 individuals to use in the next generation
- combine them (mutation, crossover, etc...)
- evolve the populations over time


- new idea:

  -

Actual Question ideas:
  - If we train a genetic algorithm for the hand to do one task, is it possible to transfer the actions learned from the genetic algorithm for the hand to do another task, and still be able to very close to it's desired goal, i.e. does it complete it's task?

  - Or, can we continue evolving populations and measure the number of generations it takes to get the 0 reward (i.e. task made), compared to the number of generations it takes normally.


  The metric we will use is the distance from the desired goal, i.e. the difference between the achieved goal and the desired goal.
