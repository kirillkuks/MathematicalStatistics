import Distributions as Dist
import matplotlib.pyplot as plt


def boxplot(distributions, sizes):
    labels = ['n = ' + str(size) for size in sizes]
    for distribution in distributions:
        samples = []
        for size in sizes:
            sample = distribution.create_sample(size)
            samples.append(sample)
        plt.boxplot(samples, vert=False, widths=0.5, labels=labels)
        plt.title(distribution.get_name())
        plt.savefig('images/boxplot/' + distribution.get_name() + '.png')
        plt.show()
    return


def share(sample, x1, x2):
    emissions_counter = 0
    for elem in sample:
        if elem < x1 or elem > x2:
            emissions_counter += 1
    return emissions_counter / len(sample)


def emissions(distribution, size, n):
    shares_means = []
    for _ in range(n):
        sample = distribution.create_sample(size)
        x1, x2 = distribution.mustache(sample)
        shares_means.append(share(sample, x1, x2))
    return distribution.mean(shares_means)


def share_of_emissions(distributions, sizes):
    n = 1000
    for distribution in distributions:
        for size in sizes:
            shares_mean = emissions(distribution, size, n)
            print(distribution.get_name() + '; n = ' + str(size) + ' | ' + str(shares_mean))


def init():
    distributions = [Dist.NormalDistribution(),
                     Dist.CauchyDistribution(),
                     Dist.LaplaceDistribution(),
                     Dist.PoissonDistribution(),
                     Dist.UniformDistribution()]
    sizes = [20, 100]
    return distributions, sizes,


def main():
    distributions, sizes = init()

    boxplot(distributions, sizes)
    share_of_emissions(distributions, sizes)

    return


if __name__ == '__main__':
    main()
