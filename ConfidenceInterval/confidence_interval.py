from normal_distribution import NormalDistribution
from numpy import sqrt
import scipy.stats as ss


class ConfidenceInterval:
    def __init__(self, sample, confidence_level: float):
        assert 0 < confidence_level < 1
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.sample = sample
        self.size = len(sample)

    def mean_confidence_interval_for_normal(self):
        mean = NormalDistribution.mean(self.sample)
        deviation = NormalDistribution.deviation(self.sample)
        left = right = mean
        t_quantile = ss.t.ppf(1 - self.alpha / 2, self.size - 1)
        shift = deviation * t_quantile / sqrt(self.size - 1)
        left -= shift
        right += shift
        return left, right

    def deviation_confidence_interval_for_normal(self):
        deviation = NormalDistribution.deviation(self.sample)
        chi_left = ss.chi2.ppf(1 - self.alpha / 2, self.size - 1)
        chi_right = ss.chi2.ppf(self.alpha / 2, self.size - 1)
        size_sqrt = sqrt(self.size)
        left = deviation * size_sqrt / sqrt(chi_left)
        right = deviation * size_sqrt / sqrt(chi_right)
        return left, right

    def mean_asymptotic_confidence_interval(self):
        mean = NormalDistribution.mean(self.sample)
        deviation = NormalDistribution.deviation(self.sample)
        left = right = mean
        size_sqrt = sqrt(self.size)
        normal_quantile = ss.norm.ppf(1 - self.alpha / 2, 0, 1)
        shift = deviation * normal_quantile / size_sqrt
        left -= shift
        right += shift
        return left, right

    def deviation_asymptotic_confidence_interval(self):
        m4 = NormalDistribution.central_moment(self.sample, 4)
        deviation = NormalDistribution.deviation(self.sample)
        normal_quantile = ss.norm.ppf(1 - self.alpha / 2)
        U = normal_quantile * sqrt((m4 / (deviation ** 4) - 1) / self.size)
        left = deviation * (1 + U) ** (-0.5)
        right = deviation * (1 - U) ** (-0.5)
        return left, right
