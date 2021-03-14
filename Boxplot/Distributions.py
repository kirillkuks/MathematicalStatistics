import numpy as np
from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def create_sample(self, size):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @staticmethod
    def mustache(sample):
        ql, qr = Distribution.quantiles(sample)
        delta = qr - ql
        x1 = ql - 3 / 2 * delta
        x2 = qr + 3 / 2 * delta
        return x1, x2

    @staticmethod
    def quantiles(sample):
        size = len(sample)
        ql = size / 4
        qr = 3 * size / 4
        if ql != int(ql):
            ql += 1
        if qr != int(qr):
            qr += 1
        sample.sort()
        return sample[int(ql)], sample[int(qr)]

    @staticmethod
    def mean(sample):
        s = 0
        for elem in sample:
            s += elem
        return s / len(sample)


class NormalDistribution(Distribution):
    def __init__(self):
        self.name = 'Normal'

    def create_sample(self, size):
        loc = 0
        scale = 1
        return np.random.normal(loc, scale, size)

    def get_name(self):
        return self.name


class CauchyDistribution(Distribution):
    def __init__(self):
        self.name = 'Cauchy'

    def create_sample(self, size):
        return np.random.standard_cauchy(size)

    def get_name(self):
        return self.name


class LaplaceDistribution(Distribution):
    def __init__(self):
        self.name = 'Laplace'

    def create_sample(self, size):
        loc = 0
        scale = 1 / (2 ** 0.5)
        return np.random.laplace(loc, scale, size)

    def get_name(self):
        return self.name


class PoissonDistribution(Distribution):
    def __init__(self):
        self.name = 'Poisson'

    def create_sample(self, size):
        lam = 10
        return np.random.poisson(lam, size)

    def get_name(self):
        return self.name


class UniformDistribution(Distribution):
    def __init__(self):
        self.name = 'Uniform'

    def create_sample(self, size):
        low = - 3 ** 0.5
        high = 3 ** 0.5
        return np.random.uniform(low, high, size)

    def get_name(self):
        return self.name
