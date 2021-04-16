import Characteristics as Char
import Distributions as Dist
import Sample


def init():
    distributions = [Dist.NormalDistribution(),
                     Dist.CauchyDistribution(),
                     Dist.LaplaceDistribution(),
                     Dist.PoissonDistribution(),
                     Dist.UniformDistribution()]
    sizes = [10, 100, 1000]
    n = 1000
    return distributions, sizes, n


def print_results(distribution, char_samples, size):
    print(distribution.get_name() + ": Size = " + str(size))
    print('Mean: &', end='')
    for sample in char_samples:
        print(' $ ' + str(round(Char.Characteristics.mean(sample.get_sample()), 4)) + ' $ ', end='&')
    print()
    print('Variance: &', end='')
    for sample in char_samples:
        print(' $ ' + str(round(Char.Characteristics.variance(sample.get_sample()), 4)) + ' $ ', end='&')
    print()
    print('Max: &', end='')
    for sample in char_samples:
        print(' $ ' + str(round(Char.Characteristics.max(sample.get_sample()), 4)) + ' $ ', end='&')
    print()
    print('Min: &', end='')
    for sample in char_samples:
        print(' $ ' + str(round(Char.Characteristics.min(sample.get_sample()), 4)) + ' $ ', end='&')
    print()
    print('\n')
    # print('Max: &', end='')
    # for sample in char_samples:
    #     print(' $ ' + str(round(Char.Characteristics.mean(sample.get_sample()) - Char.Characteristics.variance(sample.get_sample()), 4)) + ' $ ', end='&')
    # print()
    # print('Min: &', end='')
    # for sample in char_samples:
    #     print(' $ ' + str(round(Char.Characteristics.mean(sample.get_sample()) + Char.Characteristics.variance(sample.get_sample()), 3)) + ' $ ', end='&')
    # print()
    # print('\n')


def main():
    distributions, sizes, n = init()
    for distribution in distributions:
        for size in sizes:
            char_samples = [Sample.Sample() for _ in range(0, Char.Characteristics.number_of_characteristics())]
            for i in range(0, n):
                characteristics = Char.Characteristics(distribution.create_sample(size))
                chars = characteristics.get_characteristics()
                j = 0
                for char in chars:
                    char_samples[j].add_elem_to_sample(char)
                    j += 1
            print_results(distribution, char_samples, size)
    return 0


if __name__ == '__main__':
    main()
