#!python
#cython: language_level=3, boundscheck=False

import cython
import numpy
cimport numpy
import random
from three_sat.models import Solution as OutputSolution
from libc.stdlib cimport malloc, free


TOURNAMENT_TYPE_KNOCKOUT = 'knockout'
TOURNAMENT_TYPE_POOL = 'pool'

CROSSOVER_SINGLE_POINT = 'single_point'
CROSSOVER_SWAP_MAP = 'swap_map'

FITNESS_FUNCTION_SIMPLE = 'simple_fitness'
FITNESS_FUNCTION_UNSATISFIED_WEIGHT = 'finer_fitness'


_default_options = {
    'population_size': 200,
    'bit_flip_probability': 0.02,
    'bit_correction_probability': 0.25,
    'tournament_pool_size': 4,
    'tournament_win_probability': 0.95,
    'elitism_size': 15,
    'max_best_solution_age': 100,
    'max_generations': 1000,
    'no_satisfy_penalty': 5000, # Default value is never used
    'random_individuals_inserted': 5,
    'selection': TOURNAMENT_TYPE_POOL,
    'crossover': CROSSOVER_SWAP_MAP,
    'fitness': FITNESS_FUNCTION_UNSATISFIED_WEIGHT
}


cdef int satisfies_clause(numpy.ndarray[numpy.int64_t, ndim=1] clause,
                          numpy.ndarray[numpy.int8_t, ndim=1] assignment):
    cdef int i
    for i in range(len(clause)):
        if clause[i] < 0 and not assignment[-clause[i]-1]:
            return True
        elif clause[i] > 0 and assignment[clause[i]-1]:
            return True

    return False


cdef class Instance:
    cdef numpy.ndarray weights
    cdef numpy.ndarray clauses
    cdef int num_variables
    cdef int num_clauses

    def __cinit__(self, python_instance):
        self.weights = python_instance.weights
        self.clauses = python_instance.clauses
        self.num_clauses = len(python_instance.clauses)
        self.num_variables = python_instance.variables

    cpdef size(self):
        return self.num_variables


cdef class Solution:
    cdef Instance instance
    cdef numpy.ndarray assignments
    cdef int value
    cdef int unsatisfied_clauses
    cdef dict options

    def __cinit__(self, Instance instance, numpy.ndarray[numpy.int8_t, ndim=1] assignments, dict options):
        self.instance = instance
        self.assignments = assignments
        self.options = options
        self.analyze_me()


    cdef mutate_bit_correction(self):
        cdef float prob = self.options['bit_correction_probability']
        cdef float res = random.uniform(0, 1)
        while prob >= res and self.unsatisfied_clauses > 0:
            prob *= self.options['bit_correction_probability']
            self.bit_correct()
            self.analyze_me()

    cdef analyze_me(self):
        cdef int i, v, c, a, failed_0, failed_1, satisfies

        # Calculate the sum of weights of variables having been assigned 1
        self.value = 0
        for i in range(self.instance.num_variables):
            if self.assignments[i] == 1:
                self.value += self.instance.weights[i]

        # Verify that all clauses are satisfied
        self.unsatisfied_clauses = 0
        for c in range(self.instance.num_clauses):
            if not satisfies_clause(self.instance.clauses[c], self.assignments):
                self.unsatisfied_clauses += 1

    cdef bit_correct(self):
        cdef int i, c

        # Get the first unsatisfied clause and flip a random bit
        for c in range(self.instance.num_clauses):
            if not satisfies_clause(self.instance.clauses[c], self.assignments):
                i = random.randrange(0, len(self.instance.clauses[c]))
                self.assignments[i] = not self.assignments[i]
                return

    cpdef int fitness(self):
        if self.options['fitness'] == FITNESS_FUNCTION_UNSATISFIED_WEIGHT:
            if self.unsatisfied_clauses == 0:
                return self.value
            else:
                return -self.unsatisfied_clauses * self.options['no_satisfy_penalty'] + self.value
        elif self.options['fitness'] == FITNESS_FUNCTION_SIMPLE:
            if self.unsatisfied_clauses == 0:
                return self.value
            else:
                return -self.unsatisfied_clauses

    def __str__(self):
        return ' '.join([str(x) for x in self.assignments])


cdef struct RunStatistics:
    int number_of_generations


def solve(python_instance, **kwargs):

    # Overwrite the default options with ones passed in as arguments
    options = _default_options.copy()
    options.update(kwargs)

    # Convert the input instance into a Cython class
    cdef Instance instance = Instance(python_instance)

    # Set the no-satisfy penalty to be the sum of all weights
    cdef int s = 0
    for i in range(instance.num_variables):
        s += instance.weights[i]
    options['no_satisfy_penalty'] = s

    # Generate a solution and convert it to Python
    cdef RunStatistics run_statistics
    cdef Solution solution = genetic(instance, options, &run_statistics)
    return OutputSolution(python_instance, solution.assignments, solution.value, solution.unsatisfied_clauses, {
        'number_of_generations': run_statistics.number_of_generations
    })


cdef Solution genetic(Instance instance, dict options, RunStatistics* run_statistics):

    cdef int terminate = False
    cdef Solution best_solution
    cdef int generations = 1
    cdef int places_remaining, best_solution_age

    # Initialize first generation
    population = [generate_random_solution(instance, options) for i in range(options['population_size'])]
    population.sort(key=lambda s: s.fitness(), reverse=True)
    best_solution = population[0]
    best_solution_age = 0

    while not terminate:
        # Construct new generation
        new_population = []
        places_remaining = options['population_size']

        # Transfer the elite solutions
        for i in range(options['elitism_size']):
            new_population.append(population[i])
            places_remaining -= 1

        # Generate random new individuals
        for i in range(options['random_individuals_inserted']):
            new_population.append(generate_random_solution(instance, options))
            places_remaining -= 1

        # Breed new solutions
        while places_remaining > 0:
            children = breed_solutions(population, instance, options)
            new_population.extend(children)
            places_remaining -= len(children)

        # Replace the old population
        population = new_population

        # Sort the new population by fitness
        population.sort(key=lambda s: s.fitness(), reverse=True)

        # Find the best solution and remember it
        best_solution_age += 1
        new_best_solution = population[0]
        if new_best_solution.fitness() > best_solution.fitness():
            best_solution_age = 0
            best_solution = new_best_solution
        if best_solution_age >= options['max_best_solution_age']:
            terminate = True

        # Up the generation index
        generations += 1
        if generations >= options['max_generations']:
            terminate = True

    run_statistics.number_of_generations = generations - options['max_best_solution_age']

    return best_solution


cdef list breed_solutions(list population, Instance instance, dict options):
    # Select parents

    cdef Solution individual1
    cdef Solution individual2

    if options['selection'] == TOURNAMENT_TYPE_KNOCKOUT:
        individual1 = tournament_select_knockout(population, options)
        individual2 = tournament_select_knockout(population, options)

    elif options['selection'] == TOURNAMENT_TYPE_POOL:
        individual1 = tournament_select_pool(population, options)
        individual2 = tournament_select_pool(population, options)

    # Crossover
    cdef int i, tmp, crossover_point
    cdef numpy.ndarray[numpy.int8_t, ndim=1] assignments1
    cdef numpy.ndarray[numpy.int8_t, ndim=1] assignments2
    cdef numpy.ndarray[numpy.int8_t, ndim=1] crossover_map
    assignments1 = individual1.assignments.copy()
    assignments2 = individual2.assignments.copy()

    if options['crossover'] == CROSSOVER_SINGLE_POINT:
        crossover_point = random.randrange(1, instance.size() - 1)

        assignments1[crossover_point:] = individual2.assignments[crossover_point:]
        assignments2[:crossover_point] = individual1.assignments[:crossover_point]

    elif options['crossover'] == CROSSOVER_SWAP_MAP:
        crossover_map = generate_random_assignments(len(individual1.assignments))
        for i in range(len(individual1.assignments)):
            if crossover_map[i] == 1:
                tmp = assignments1[i]
                assignments1[i] = assignments2[i]
                assignments2[i] = tmp

    # Bit-flip mutate
    mutate_assignments(assignments1, options)
    mutate_assignments(assignments2, options)

    # Construct solutions (with optional bit-correction)
    individual1 = Solution(instance, assignments1, options)
    individual1.mutate_bit_correction()
    individual2 = Solution(instance, assignments2, options)
    individual2.mutate_bit_correction()

    return [individual1, individual2]



cdef mutate_assignments(numpy.ndarray[numpy.int8_t, ndim=1] assignments, dict options):
    cdef int i, j, k, tmp
    cdef float res, prob

    # Bit flip
    for i in range(len(assignments)):
        res = random.uniform(0, 1)
        if options['bit_flip_probability'] >= res:
            assignments[i] = not assignments[i]


cdef Solution tournament_select_knockout(list population, dict options):
    cdef int i
    cdef float res
    cdef int pool_size = options['tournament_pool_size']
    cdef Solution champion = population[random.randrange(0,len(population))]
    cdef Solution challenger

    for i in range(pool_size):
        challenger = population[random.randrange(0,len(population))]
        if challenger.fitness() > champion.fitness():
            champion = challenger

    res = random.uniform(0, 1)
    if options['tournament_win_probability'] >= res:
        return champion
    else:
        return challenger


cdef Solution tournament_select_pool(list population, dict options):
    cdef int i
    cdef float res
    cdef list tournament_pool = [population[random.randrange(0,len(population))] for i in range(options['tournament_pool_size'])]
    tournament_pool.sort(key=lambda s: s.fitness(), reverse=True)

    res = random.uniform(0, 1)
    if options['tournament_win_probability'] >= res or len(tournament_pool) <= 1:
        return tournament_pool[0]
    else:
        return tournament_pool[1]


cdef Solution generate_random_solution(Instance instance, dict options):
    cdef Solution solution = Solution(instance, generate_random_assignments(instance.num_variables), options)
    return solution


cdef numpy.ndarray[numpy.int8_t, ndim=1] generate_random_assignments(int size):
    cdef numpy.ndarray[numpy.int8_t, ndim=1] assignments
    assignments = numpy.random.choice(numpy.array([0, 1], dtype=numpy.int8), size)
    return assignments
