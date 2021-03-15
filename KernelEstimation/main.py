import Distributions as Dist
import numpy as np
import matplotlib.pyplot as plt


def init():
    distributions = [Dist.NormalDistribution(-4, 4),
                     Dist.CauchyDistribution(-4, 4),
                     Dist.LaplaceDistribution(-4, 4),
                     Dist.PoissonDistribution(6, 14),
                     Dist.UniformDistribution(-4, 4)]
    sizes = [20, 60, 100]
    return distributions, sizes


def _edf(sample, x):
    inc = 1 / len(sample)
    y = 0
    for elem in sample:
        if elem < x:
            y += inc
            continue
        return y
    return y


def edf(sample, x):
    y = []
    sample.sort()
    for elem in x:
        y.append(_edf(sample, elem))
    return y


def empirical_distribution_function(distributions, sizes):
    for distribution in distributions:
        for size in sizes:
            sample = distribution.create_sample(size)
            a, b = distribution.get_section()

            x = np.linspace(a, b, 100)
            y = distribution.cumulative(x)
            ey = edf(sample, x)
            plt.title(distribution.get_name() + "; size = " + str(size))

            plt.plot(x, ey)
            plt.plot(x, y)

            plt.savefig('images/edf/' + 'EDF' + distribution.get_name() + str(size) + '.png')
            plt.show()


def _edenf(sample, x, hn):
    y = 0
    for elem in sample:
        y += Dist.Distribution.kernel((x - elem) / hn)
    return y / (len(sample) * hn)


def edenf(sample, x, hn):
    y = []
    for elem in x:
        y.append(_edenf(sample, elem, hn))
    return y


def empirical_density_function(distributions, sizes):
    for distribution in distributions:
        for size in sizes:
            sample = distribution.create_sample(size)
            hn = distribution.hn(sample) / 2
            a, b = distribution.get_section()

            x = np.linspace(a, b, 100)
            y = distribution.density(x)
            for k in range(3):
                ey = edenf(sample, x, hn)

                plt.title(distribution.get_name() + "; size = " + str(size))
                axes = plt.gca()
                axes.set_ylim([0, 1])
                plt.plot(x, ey)
                plt.plot(x, y)
                hn *= 2
                plt.savefig('images/den/' + 'DENh' + str(k) + distribution.get_name() + str(size))
                plt.show()


def main():
    distributions, sizes = init()

    # empirical_distribution_function(distributions, sizes)
    empirical_density_function(distributions, sizes)

    return


if __name__ == '__main__':
    main()
