from numpy.random import default_rng


class Correlation:
    def __init__(self):
        self.rng = default_rng()

    def multivariate_normal(self, mean, variance, correlation, size):
        cov = [[variance[0], correlation],
               [correlation, variance[1]]]
        return self.rng.multivariate_normal(mean, cov, size).T

    def mixed_multivariate_normal(self, size):
        mean = [0, 0]
        cov1 = [[1, 0.9],
                [0.9, 1]]
        cov2 = [[10, -0.9],
                [-0.9, 10]]
        return 0.9 * self.rng.multivariate_normal(mean, cov1, size).T +\
            0.1 * self.rng.multivariate_normal(mean, cov2, size).T

    @staticmethod
    def mean(sample):
        s = 0
        for elem in sample:
            s += elem
        return s / len(sample)

    @staticmethod
    def median(sample):
        sam = sample.copy()
        sam.sort()
        size = len(sam)
        half = size // 2
        if size % 2:
            return sam[half + 1]
        return (sam[half] + sam[half + 1]) / 2

    @staticmethod
    def square_mean(sample):
        s = 0
        for elem in sample:
            s += elem * elem
        return s / len(sample)

    @staticmethod
    def variance(sample):
        mean = Correlation.mean(sample)
        s = 0
        for elem in sample:
            s += (elem - mean)**2
        return s / len(sample)

    @staticmethod
    def pearson_correlation(x, y):
        size = len(x)
        assert size == len(y)

        x_mean = Correlation.mean(x)
        y_mean = Correlation.mean(y)
        num = 0
        s1, s2 = 0, 0
        for i in range(0, size):
            delta_x = x[i] - x_mean
            delta_y = y[i] - y_mean
            num += delta_x * delta_y
            s1 += delta_x * delta_x
            s2 += delta_y * delta_y
        norm_cof = 1 / size
        return norm_cof * num / (norm_cof * s1 * norm_cof * s2)**(1 / 2)

    @staticmethod
    def square_correlation(x, y):
        size = len(x)
        assert size == len(y)

        med_x = Correlation.median(x)
        med_y = Correlation.median(y)
        n1, n2, n3, n4 = 0, 0, 0, 0
        for i in range(0, size):
            if x[i] - med_x >= 0:
                if y[i] - med_y >= 0:
                    n1 += 1
                else:
                    n4 += 1
            else:
                if y[i] - med_y >= 0:
                    n2 += 1
                else:
                    n3 += 1
        return ((n1 + n3) - (n2 + n4)) / size

    @staticmethod
    def _ranking(sample):
        size = len(sample)
        pairs = [(sample[i], i) for i in range(0, size)]
        pairs = sorted(pairs, key=lambda p: p[0])
        ranks = [0 for _ in range(0, size)]
        cur_rank = 1
        for pair in pairs:
            ranks[pair[1]] = cur_rank
            cur_rank += 1
        return ranks

    @staticmethod
    def spearman_correlation(x, y):
        size = len(x)
        assert size == len(y)

        u = Correlation._ranking(x)
        v = Correlation._ranking(y)
        return Correlation.pearson_correlation(u, v)
