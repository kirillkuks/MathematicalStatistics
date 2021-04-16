import distribution as dist
from numpy import log10


class Hypothesis:
    def __init__(self):
        self.chi_quantiles = [3.8415, 5.9915, 7.8147, 9.4877, 11.0705,
                              12.5916, 14.0671, 15.5073, 16.9190, 18.3070]

    @staticmethod
    def _get_frequencies(sample, a: float, b: float, k: int):
        frequencies = [0 for _ in range(k)]
        delta = (b - a) / k
        for elem in sample:
            k_i = int((elem - a) / delta)
            frequencies[k_i] += 1
        return frequencies

    @staticmethod
    def _calculate_quantile(frequencies: [], probabilities: [], size: int):
        k = len(frequencies)
        assert k == len(probabilities)
        quantile = 0
        for i in range(k):
            quantile += ((frequencies[i] - size * probabilities[i]) ** 2) / (size * probabilities[i])
        return quantile

    def check_hypothesis(self, sample, distribution: dist.Distribution):
        size = len(sample)
        k = Hypothesis.number_of_sections(size)
        a, b = min(sample), max(sample)
        delta = (b - a) / k
        a, b = a - delta / 2, b + delta / 2
        frequencies = Hypothesis._get_frequencies(sample, a, b, k)
        probabilities = distribution.get_probabilities(a, b, k)
        quantile = self.chi_quantiles[k - 2]
        sample_quantile = Hypothesis._calculate_quantile(frequencies, probabilities, size)
        print(quantile)
        print(sample_quantile)
        return

    @staticmethod
    def number_of_sections(size: int):
        return int(1 + 3.3 * log10(size))
