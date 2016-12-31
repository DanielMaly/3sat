import cython
import numpy
cimport numpy
from three_sat.models import Solution as OutputSolution, Instance as PythonInstance

_default_options = {
    'population_size': 50,
    'mutation_probability': 0.02,
    'tournament_pool_size': 4,
    'tournament_win_probability': 0.9,
    'elitism_size': 7,
    'maximum_solution_age': 100,
    'no_satisfy_penalty': 5000
}


@cython.boundscheck(False)
cdef int clause_contains(int v, numpy.ndarray[numpy.int64_t, ndim=1] clause):
    cdef int i

    for i in range(len(clause)):
        if clause[i] == v:
            return True
        elif abs(clause[i]) > abs(v):
            return False

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


cdef class Solution:
    cdef Instance instance
    cdef numpy.ndarray assignments
    cdef int value
    cdef int satisfies
    cdef dict options

    def __cinit__(self, Instance instance, numpy.ndarray[numpy.int8_t, ndim=1] assignments, dict options):
        self.instance = instance
        self.assignments = assignments
        self.options = options
        self.analyze_me()

    @cython.boundscheck(False)
    cdef analyze_me(self):
        cdef int i, v, c, a, failed_0, failed_1, satisfies

        # Calculate the sum of weights of variables having been assigned 1
        self.value = 0
        self.satisfies = True
        for i in range(self.instance.num_variables):
            if self.assignments[i] == 1:
                self.value += self.instance.weights[i]

            for c in range(self.instance.num_clauses):
                failed_1 = self.assignments[i] and clause_contains(-i, self.instance.clauses[c])
                failed_0 = not self.assignments[i] and clause_contains(i, self.instance.clauses[c])
                if failed_0 or failed_1:
                    self.satisfies = False
                    break

    cdef int fitness(self):
        cdef int fitness = self.value
        if not self.satisfies:
            fitness -= self.options['no_satisfy_penalty']
        return fitness


def solve(python_instance, **kwargs):

    # Overwrite the default options with ones passed in as arguments
    options = _default_options.copy()
    options.update(kwargs)

    cdef Instance instance = Instance(python_instance)
    cdef Solution solution = genetic(instance, options)
    return OutputSolution(python_instance, solution.assignments, solution.value)


cdef Solution genetic(Instance instance, dict options):
    cdef numpy.ndarray[numpy.int8_t, ndim=1] assignments = numpy.random.choice(numpy.array([0, 1], dtype=numpy.int8), instance.num_variables)
    cdef Solution solution = Solution(instance, assignments, options)
    return solution