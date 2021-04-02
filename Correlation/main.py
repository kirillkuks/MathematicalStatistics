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
                ellipse = Ellipse(x, y)
                ellipse.plot()
            print_correlation(correlation_pearson_sample, 'Pearson', size, rho)


def ellipses(sizes, correlations, mean, variance):
    correlation = Correlation()
    for size in sizes:
        for rho in correlations:
            x, y = correlation.multivariate_normal(mean, variance, rho, size)
            ellipse = Ellipse(x, y, size, rho)
            ellipse.plot()


def main():
    sizes, n, correlations, mean, variance = init()
    selective_correlation(sizes, n, correlations, mean, variance)
    ellipses(sizes, correlations, mean, variance)
    return 0


if __name__ == '__main__':
    Correlation.spearman_correlation([4, 5, 1, 3, 5, 11], [5, 6, 1, 5, 6, 6])
    main()