from numpy.random import default_rng


class Generator:
    def __init__(self, function, section):
        assert len(section) == 2
        assert section[0] < section[1]

        self.function = function
        self.section = section
        self.rng = default_rng()

    def create_samples(self, step_num):
        step = (self.section[1] - self.section[0]) / (step_num - 1)
        x = [self.section[0] + i * step for i in range(0, step_num)]
        errors = self.rng.standard_normal(step_num)
        y = self.function(x) + errors
        return x, y

    def create_sample_with_fluctuations(self, step_num, fluctuations):
        assert len(fluctuations) == 2
        x, y = self.create_samples(step_num)
        y[0] += fluctuations[0]
        y[step_num - 1] += fluctuations[1]
        return x, y
