import distribution as dist
from characteristics import Characteristics
from hypothesis import Hypothesis


def normal_hypothesis(distribution: dist.Distribution, size: int):
    sample = distribution.create_sample(size)
    characteristics = Characteristics(sample)
    hypothesis = Hypothesis()
    hyp_distribution = dist.NormalDistribution(characteristics.mean(), characteristics.variance())
    hypothesis.check_hypothesis(sample, hyp_distribution)


def laplace_hypothesis(distribution: dist.Distribution, size: int):
    sample = distribution.create_sample(size)
    characteristics = Characteristics(sample)
    hypothesis = Hypothesis()
    hyp_distribution = dist.LaplaceDistribution(characteristics.mean(), characteristics.variance() / (2 ** 0.5))
    hypothesis.check_hypothesis(sample, hyp_distribution)


def uniform_hypothesis(distribution: dist.Distribution, size: int):
    sample = distribution.create_sample(size)
    characteristics = Characteristics(sample)
    hypothesis = Hypothesis()
    hyp_distribution = dist.UniformDistribution(characteristics.min(), characteristics.max())
    hypothesis.check_hypothesis(sample, hyp_distribution)


def main():
    size = 100
    normal_distribution = dist.NormalDistribution(0, 1)
    print('Normal for Normal')
    normal_hypothesis(normal_distribution, size)

    size = 20

    laplace_distribution = dist.LaplaceDistribution(0, 1 / (2 ** 0.5))
    uniform_distribution = dist.UniformDistribution(- 3 ** 0.5, 3 ** 0.5)

    print('Laplace for Normal')
    normal_hypothesis(laplace_distribution, size)
    print('Uniform for Normal')
    normal_hypothesis(uniform_distribution, size)

    print('Normal for Laplace')
    laplace_hypothesis(normal_distribution, size)
    print('Normal for Uniform')
    uniform_hypothesis(normal_distribution, size)

    return 0


if __name__ == '__main__':
    main()
