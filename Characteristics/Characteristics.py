import numpy as np


class Characteristics:
    def __init__(self, sample):
        sample.sort()
        self.sample = sample.tolist()
        self.size = len(sample)

    def sample_mean(self):
        return Characteristics.mean(self.sample)

    def median(self):
        half = self.size // 2
        if self.size % 2:
            return self.sample[half + 1]
        return (self.sample[half] + self.sample[half + 1]) / 2

    def extremum_half_sum(self):
        return (self.sample[0] + self.sample[self.size - 1]) / 2

    def quantile_half_sum(self):
        ql = self.size / 4
        qr = 3 * self.size / 4
        if ql != np.floor(ql):
            ql += 1
        if qr != np.floor(qr):
            qr += 1
        return (self.sample[int(ql)] + self.sample[int(qr)]) / 2

    def truncated_mean(self):
        r = self.size // 4
        mean = 0
        for i in range(r, self.size - r):
            mean += self.sample[i]
        return mean / (self.size - 2 * r)

    def get_characteristics(self):
        yield self.sample_mean()
        yield self.median()
        yield self.extremum_half_sum()
        yield self.quantile_half_sum()
        yield self.truncated_mean()

    @staticmethod
    def number_of_characteristics():
        return 5

    @staticmethod
    def mean(sample):
        mean = 0
        for x in sample:
            mean += x
        return mean / len(sample)

    @staticmethod
    def variance(sample):
        mean_square = Characteristics.mean([x * x for x in sample])
        square_mean = Characteristics.mean(sample)
        return mean_square - square_mean * square_mean

    @staticmethod
    def max(sample):
        return max(sample)

    @staticmethod
    def min(sample):
        return min(sample)
