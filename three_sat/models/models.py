import re
import numpy
import os


class Instance:
    def __init__(self, weights, clauses, identifier=None):
        self.weights = weights
        self.clauses = clauses
        self.identifier = identifier
        self.variables = len(weights)

    def __str__(self, *args, **kwargs):
        first_line = "p cnf {} {}\n".format(self.variables, len(self.clauses))
        weight_line = ' '.join([str(int(w)) for w in self.weights])
        return first_line + '\n'.join(Instance.serialize_clause(c) for c in self.clauses) + '\n' + weight_line

    @classmethod
    def serialize_clause(cls, c):
        return ' '.join([str(a) for a in c]) + ' 0'

    @classmethod
    def parse_clause(cls, s):
        splitzed = s.split(' ')
        return [int(w) for w in splitzed][:-1]

    @classmethod
    def from_file(cls, filename):
        with open(filename, mode='r') as in_file:
            initial_line_pattern = re.compile('p cnf (\d+) (\d+)')
            match = initial_line_pattern.match(in_file.readline())
            n_clauses = int(match.group(2))

            clauses = numpy.array([cls.parse_clause(in_file.readline()) for i in range(n_clauses)])

            last_line = in_file.readline()
            weight_strings = last_line.split(' ')
            weights = numpy.array([int(w) for w in weight_strings])

            identifier = os.path.basename(filename).split('.inst.dat')[0]
            return cls(weights, clauses, identifier)


class Solution:
    def __init__(self, instance, assignments, value, unsatisfied, statistics):
        self.instance = instance
        self.assignments = assignments
        self.value = value
        self.statistics = statistics
        self.unsatisfied = unsatisfied

    def fitness(self):
        if self.unsatisfied == 0:
            return self.value
        else:
            return -self.unsatisfied

    def __str__(self, *args, **kwargs):
        assignments = ' '.join([str(x) for x in self.assignments])
        return ' | '.join([self.instance.identifier, assignments, str(self.value),
                           str(self.fitness()),
                           str(self.statistics['number_of_generations'])])
