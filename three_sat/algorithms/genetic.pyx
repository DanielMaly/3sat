import numpy
cimport numpy
from three_sat.models import Solution as OutputSolution, Instance as PythonInstance

_default_options = {
    'population_size': 50,
    'mutation_probability': 0.02,
    'tournament_pool_size': 4,
    'tournament_win_probability': 0.9,
    'maximum_solution_age': 100,
    'no_satisfy_penalty': 5000
}


cdef class Instance:
    cdef numpy.ndarray[numpy.int8_t, ndim=1] weights
    cdef numpy.ndarray[numpy.int64_t, ndim=2] clauses
    cdef int num_variables
    cdef int num_clauses

    cdef __cinit__(self, PythonInstance instance):
        pass


cdef class Solution:
    cdef Instance instance
    cdef numpy.ndarray[numpy.int8_t, ndim=1] assignments
    cdef int value
    cdef int satisfies
    cdef dict options

    cdef __cinit__(self, Instance instance, numpy.ndarray[numpy.int8_t, ndim=1] assignments, dict options):
        self.instance = instance
        self.assignments = assignments
        self.options = options

        # Calculate the sum of weights of variables having been assigned 1
        self.value = 0
        for i in range(self.instance.num_variables):
            if self.assignments[i] == 1:
                self.value += self.instance.weights[i]

        # Determine if the solution satisfies the 3SAT formula
        satisfies = 1


    cdef int fitness(self):
        cdef int fitness = self.value
        if not self.satisfies:
            fitness -= self.options['no_satisfy_penalty']
        return fitness



cpdef OutputSolution solve(PythonInstance instance, **kwargs):

    # Overwrite the default options with ones passed in as arguments
    options = _default_options.copy()
    options.update(kwargs)

    assignments = numpy.random.choice([0, 1], instance.variables)
    solution = OutputSolution(instance, assignments, 50)

    return solution


cdef Solution genetic(Instance instance, dict options):
    cdef Solution solution = Solution()
    return solution