class Instance:
    def __init__(self, weights, clausules):
        self.weights = weights
        self.clausules = clausules

    def __str__(self, *args, **kwargs):
        first_line = "p cnf {} {}\n".format(len(self.weights), len(self.clausules))
        weight_line = ' '.join([str(int(w)) for w in self.weights])
        return first_line + '\n'.join(Instance.serialize_clausule(c) for c in self.clausules) + '\n' + weight_line

    @classmethod
    def serialize_clausule(cls, c):
        return ' '.join([str(a) for a in c]) + ' 0'


class Solution:
    pass