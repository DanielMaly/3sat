from models import Instance
import numpy
import math


def generate_instance(variables, ratio, weight_min, weight_max):
    n_clausules = math.floor(ratio * variables)
    weights = numpy.around(numpy.random.uniform(weight_min, weight_max, variables))
    clausules = numpy.array([generate_clausule(variables) for i in range(n_clausules)])
    return Instance(weights, clausules)


def generate_clausule(variables, num_literals=3):
    literals = numpy.sort(numpy.random.choice(variables, num_literals, replace=False) + 1)
    return literals * numpy.random.choice([-1, 1], num_literals)
