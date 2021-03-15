import numpy as np
import scipy.special as ss
import scipy.stats as sst
from abc import ABC, abstractmethod


class Distribution(ABC):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get_section(self):
        return self.a, self.b

    @abstractmethod
    def create_sample(self, size):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def cumulative(self, x):
        pass

    @abstractmethod
    def density(self, x):
        pass

    @staticmethod
    def mean(sample):
        s = 0
        for elem in sample:
            s += elem
        return s / len(sample)

    @staticmethod
    def var(sample):
        var = 0
        mean = Distribution.mean(sample)
        for elem in sample:
            var += (elem - mean) ** 2
        return var / len(sample)

    @staticmethod
    def hn(sample):
        var = Distribution.var(sample)
        deviation = np.sqrt(var)
        return 1.06 * deviation * (len(sample) ** (- 1 / 5))

    @staticmethod
    def kernel(u):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-(u ** 2) / 2)


class NormalDistribution(Distribution):
    def __init__(self, a, b):
        Distribution.__init__(self, a, b)
        self.name = 'Normal'

    def create_sample(self, size):
        loc = 0
        scale = 1
        return np.random.normal(loc, scale, size)

    def cumulative(self, x):
        return 1 / 2 * (1 + ss.erf(x / np.sqrt(2)))

    def density(self, x):
        return 1 / ((2 * np.pi) ** 0.5) * np.exp(-(x ** 2) / 2)

    def get_name(self):
        return self.name


class CauchyDistribution(Distribution):
    def __init__(self, a, b):
        Distribution.__init__(self, a, b)
        self.name = 'Cauchy'

    def create_sample(self, size):
        return np.random.standard_cauchy(size)

    def cumulative(self, x):
        return 1 / np.pi * np.arctan(x) + 1 / 2

    def density(self, x):
        return (1 / np.pi) * (1 / (x ** 2 + 1))

    def get_name(self):
        return self.name


class LaplaceDistribution(Distribution):
    def __init__(self, a, b):
        Distribution.__init__(self, a, b)
        self.name = 'Laplace'

    def create_sample(self, size):
        loc = 0
        scale = 1 / (2 ** 0.5)
        return np.random.laplace(loc, scale, size)

    def cumulative(self, x):
        a = 1 / np.sqrt(2)
        return [1 / 2 * np.exp(a * elem) if elem <= 0 else 1 - 1 / 2 * np.exp(- a * elem) for elem in x]

    def density(self, x):
        return 1 / (2 ** 0.5) * np.exp(-(2 ** 2) * np.abs(x))

    def get_name(self):
        return self.name


class PoissonDistribution(Distribution):
    def __init__(self, a, b):
        Distribution.__init__(self, a, b)
        self.name = 'Poisson'

    def create_sample(self, size):
        lam = 10
        return np.random.poisson(lam, size)

    def cumulative(self, k):
        return sst.poisson.cdf(k, 10)

    def density(self, k):
        return [(10 ** int(elem)) / np.math.factorial(int(elem)) * np.exp(-10) for elem in k]

    def get_name(self):
        return self.name


class UniformDistribution(Distribution):
    def __init__(self, a, b):
        Distribution.__init__(self, a, b)
        self.name = 'Uniform'

    def create_sample(self, size):
        low = - 3 ** 0.5
        high = 3 ** 0.5
        return np.random.uniform(low, high, size)

    def cumulative(self, x):
        edge = 3 ** 0.5
        a = -edge
        b = edge
        y = []
        for elem in x:
            if elem < a:
                y.append(0)
            elif a <= elem < b:
                y.append((elem - a) / (b - a))
            else:
                y.append(1)
        return y

    def density(self, x):
        value = 1 / (2 * (3 ** 0.5))
        edge = 3 ** 0.5
        return [value if -edge <= elem <= edge else 0 for elem in x]

    def get_name(self):
        return self.name
