import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage.filters import gaussian_filter1d
from abc import ABC, abstractmethod


def histogram(sample):
    number_of_intervals = math.floor(1.72 * (sample.size ** (1 / 3)))
    sample.sort()
    left = sample[0]
    right = sample[sample.size - 1]
    delta = (right - left) / number_of_intervals
    left -= delta / 2
    right += delta / 2
    delta = (right - left) / number_of_intervals
    data = [0 for i in range(0, number_of_intervals)]

    i = 0
    for elem in sample:
        if not (left + i * delta <= elem < left + (i + 1) * delta):
            while not (left + i * delta <= elem < left + (i + 1) * delta):
                i += 1
        data[i] += 1

    for i in range(0, number_of_intervals):
        data[i] /= sample.size * delta

    bins = [left + i * delta for i in range(0, number_of_intervals)]

    return [left, bins, right]


class Distribution(ABC):
    @abstractmethod
    def create_sample(self, size):
        pass

    @abstractmethod
    def density(self, x):
        pass

    @abstractmethod
    def get_name(self):
        pass


class NormalDistribution(Distribution):
    def __init__(self):
        self.name = 'Normal'

    def create_sample(self, size):
        loc = 0
        scale = 1
        return np.random.normal(loc, scale, size)

    def density(self, x):
        return 1 / ((2 * np.pi) ** 0.5) * np.exp(-(x ** 2) / 2)

    def get_name(self):
        return self.name


class CauchyDistribution(Distribution):
    def __init__(self):
        self.name = 'Cauchy'

    def create_sample(self, size):
        return np.random.standard_cauchy(size)

    def density(self, x):
        return 1 / np.pi * (1 / (x ** 2 + 1))

    def get_name(self):
        return self.name


class LaplaceDistribution(Distribution):
    def __init__(self):
        self.name = 'Laplace'

    def create_sample(self, size):
        loc = 0
        scale = 1 / (2 ** 0.5)
        return np.random.laplace(loc, scale, size)

    def density(self, x):
        return 1 / (2 ** 0.5) * np.exp(-(2 ** 2) * np.abs(x))

    def get_name(self):
        return self.name


class PoissonDistribution(Distribution):
    def __init__(self):
        self.name = 'Poisson'

    def create_sample(self, size):
        lam = 10
        return np.random.poisson(lam, size)

    def density(self, k):
        return [(10 ** int(elem)) / np.math.factorial(int(elem)) * np.exp(-10) for elem in k]

    def get_name(self):
        return self.name


class UniformDistribution(Distribution):
    def __init__(self):
        self.name = 'Uniform'

    def create_sample(self, size):
        low = - 3 ** 0.5
        high = 3 ** 0.5
        return np.random.uniform(low, high, size)

    def density(self, x):
        value = 1 / (2 * (3 ** 0.5))
        edge = 3 ** 0.5
        return [value if -edge <= elem <= edge else 0 for elem in x]

    def get_name(self):
        return self.name


def main():
    distributions = [NormalDistribution(),
                     CauchyDistribution(),
                     LaplaceDistribution(),
                     PoissonDistribution(),
                     UniformDistribution()]
    sizes = [10, 50, 1000]
    for distribution in distributions:
        for size in sizes:
            sample = distribution.create_sample(size)
            left, bins, right = histogram(sample)
            plt.hist(sample, bins, density=True)
            x = np.linspace(left, right, 100)
            y = gaussian_filter1d(distribution.density(x), sigma=3)
            plt.plot(x, y, '--')
            plt.title(distribution.get_name() + '; Size = ' + str(size))
            plt.show()


if __name__ == '__main__':
    main()
