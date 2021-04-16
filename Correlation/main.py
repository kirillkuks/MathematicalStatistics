from correlation import Correlation
from ellipse import Ellipse


def init():
    sizes = [20, 60, 100]
    n = 1000
    correlations = [0, 0.5, 0.9]
    mean = [0, 0]
    variance = [1, 1]
    return sizes, n, correlations, mean, variance


def print_correlation(sample, name, size, rho):
    print(name + ', ' + 'size = ' + str(size) + ', ' + 'rho = ' + str(rho))
    print(Correlation.mean(sample))
    print(Correlation.square_mean(sample))
    print(Correlation.variance(sample))
    print()


def print_correlations(samples, size, rho):
    print('size = ' + str(size) + ', rho = ' + str(rho))
    for sample in samples:
        print(' $ ' + str(round(Correlation.mean(sample), 4)) + ' $ ', end='&')
    print()
    for sample in samples:
        print(' $ ' + str(round(Correlation.square_mean(sample), 4)) + ' $ ', end='&')
    print()
    for sample in samples:
        print(' $ ' + str(round(Correlation.variance(sample), 4)) + ' $ ', end='&')
    print()


def selective_correlation(sizes, n, correlations, mean, variance):
    correlation = Correlation()
    for size in sizes:
        for rho in correlations:
            correlation_pearson_sample = []
            correlation_square_sample = []
            correlation_spearman_sample = []
            for _ in range(0, n):
                x, y = correlation.multivariate_normal(mean, variance, rho, size)
                correlation_pearson_sample.append(Correlation.pearson_correlation(x, y))
                correlation_square_sample.append(Correlation.square_correlation(x, y))
            correlation_spearman_sample.append(Correlation.spearman_correlation(x, y))
            print_correlations([correlation_pearson_sample,
                                correlation_spearman_sample,
                                correlation_square_sample],
                               size, rho)
    for size in sizes:
        correlation_pearson_sample = []
        correlation_square_sample = []
        correlation_spearman_sample = []
        for _ in range(0, n):
            x, y = correlation.mixed_multivariate_normal(size)
            correlation_pearson_sample.append(Correlation.pearson_correlation(x, y))
            correlation_square_sample.append(Correlation.square_correlation(x, y))
            correlation_spearman_sample.append(Correlation.spearman_correlation(x, y))
        print_correlations([correlation_pearson_sample,
                            correlation_spearman_sample,
                            correlation_square_sample],
                           size, -1)


def ellipses(sizes, correlations, mean, variance):
    correlation = Correlation()
    for size in sizes:
        for rho in correlations:
            x, y = correlation.multivariate_normal(mean, variance, rho, size)
            ellipse = Ellipse(x, y, size, rho)
            ellipse.plot()


def main():
    sizes, n, correlations, mean, variance = init()
    # selective_correlation(sizes, n, correlations, mean, variance)
    ellipses(sizes, correlations, mean, variance)
    return 0


if __name__ == '__main__':
    main()
