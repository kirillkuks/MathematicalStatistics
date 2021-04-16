import numpy as np


class Regression:
    def __init__(self, x, y):
        assert len(x) == len(y)

        self.x = x
        self.y = y
        self.size = len(x)

    @staticmethod
    def mean(sample):
        s = 0
        for elem in sample:
            s += elem
        return s / len(sample)

    @staticmethod
    def variance(sample):
        mean = Regression.mean(sample)
        s = 0
        for elem in sample:
            s += (elem - mean)**2
        return s / len(sample)

    @staticmethod
    def median(sample):
        size = len(sample)
        half = size // 2
        if size % 2:
            return sample[half + 1]
        return (sample[half] + sample[half + 1]) / 2

    def least_squares_regression(self):
        mean_x = Regression.mean(self.x)
        mean_y = Regression.mean(self.y)
        mean_xy = Regression.mean([self.x[i] * self.y[i] for i in range(0, self.size)])
        mean_x2 = Regression.mean([elem * elem for elem in self.x])
        a = (mean_xy - mean_x * mean_y) / (mean_x2 - mean_x * mean_x)
        b = mean_y - mean_x * a
        return a, b

    def least_absolute_regression(self):
        copy_x = self.x.copy()
        copy_y = self.y.copy()
        copy_x.sort()
        copy_y.sort()
        med_x = Regression.median(copy_x)
        med_y = Regression.median(copy_y)
        rq = 0
        for i in range(0, self.size):
            rq += np.sign(self.x[i] - med_x) * np.sign(self.y[i] * med_y)
        rq /= self.size

        qw = self.size / 4
        l = int(qw) if np.abs(qw - int(qw)) < 1 / 10 else int(qw) + 1
        j = self.size - l + 1
        qx = (copy_x[j] - copy_x[l]) * np.sqrt(Regression.variance(copy_x))
        qy = (copy_y[j] - copy_y[l]) * np.sqrt(Regression.variance(copy_y))
        a = rq * (qy / qx)
        b = med_y - a * med_x
        return a, b

