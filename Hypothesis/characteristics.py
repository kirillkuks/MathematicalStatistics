class Characteristics:
    def __init__(self, sample):
        self.sample = sample
        self.size = len(sample)

    def mean(self):
        s = 0
        for elem in self.sample:
            s += elem
        return s / self.size

    def variance(self):
        mean = self.mean()
        s = 0
        for elem in self.sample:
            s += (elem - mean) ** 2
        return s / self.size

    def max(self):
        return max(self.sample)

    def min(self):
        return min(self.sample)
