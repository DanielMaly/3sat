import math

import numpy

from three_sat.models import Instance


def generate_instance(variables, ratio, weight_min, weight_max):
    n_clauses = math.floor(ratio * variables)
    weights = numpy.around(numpy.random.uniform(weight_min, weight_max, variables))
    clauses = numpy.array([generate_clause(variables) for i in range(n_clauses)])
    return Instance(weights, clauses)


def generate_clause(variables, num_literals=3):
    literals = numpy.sort(numpy.random.choice(variables, num_literals, replace=False) + 1)
    return literals * numpy.random.choice([-1, 1], num_literals)
