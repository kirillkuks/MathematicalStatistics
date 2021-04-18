from numpy.random import default_rng
from numpy import sqrt


class NormalDistribution:
    def __init__(self, loc: float, scale: float):
        self.rng = default_rng()
        self.loc = loc
        self.scale = scale

    def create_sample(self, size):
        return self.rng.normal(self.loc, self.scale, size)

    @staticmethod
    def mean(sample):
        s = 0
        for elem in sample:
            s += elem
        return s / len(sample)

    @staticmethod
    def deviation(sample):
        mean = NormalDistribution.mean(sample)
        s = 0
        for elem in sample:
            s += (elem - mean) ** 2
        return sqrt(s / len(sample))

    @staticmethod
    def central_moment(sample, order: int):
        assert order > 0
        mean = NormalDistribution.mean(sample)
        s = 0
        for elem in sample:
            s += (elem - mean) ** order
        return s / len(sample)
