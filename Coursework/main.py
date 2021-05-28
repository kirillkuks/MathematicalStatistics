import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA
from hypothesis import Hypothesis
from analyzer import Analyzer
from edf import empirical_density_function


def chars(an: Analyzer):
    size = len(an.get_data()[0])
    for i in range(1, size):
        print('#########################################\n')
        print(an.get_axis_name(i))
        axis_data = an.get_axis_data(i)
        print(np.mean(axis_data))
        print(np.var(axis_data))
        print(np.min(axis_data))
        print(np.max(axis_data))


def hist(an: Analyzer):
    size = len(an.get_data()[0])
    for i in range(1, size):
        axis_data = an.get_axis_data(i)
        num_bins = Analyzer.bins_num(len(axis_data))
        plt.hist(axis_data, num_bins, density=True)
        plt.title(an.get_axis_name(i))
        plt.show()


def boxplot(an: Analyzer):
    size = len(an.get_data()[0])
    for i in range(1, size):
        axis_data = an.get_axis_data(i)
        plt.boxplot(axis_data, vert=False, widths=0.5)
        plt.title(an.get_axis_name(i))
        plt.show()


def pca(an: Analyzer, n_components: int):
    # x = an.get_data()
    # pca_ = PCA(n_components=n_components)
    # x_pca = pca_.fit_transform(x)
    # plt.plot(x_pca)
    # print(x_pca)
    colors = ['g', 'y', 'b']
    for i in range(0, 3):
        data = an.get_one(i)
        pca_ = IncrementalPCA(n_components=n_components)
        x_pca = pca_.fit_transform(data)
        print(len(x_pca))
        x, y = [], []
        for elem in x_pca:
            x.append(elem[0])
            y.append(elem[1])
        plt.plot(x, y, 'o', color=colors[i])
    plt.show()
    return


def correlation_matrix(an: Analyzer):
    size = len(an.get_data()[0])
    for i in range(0, size):
        print(an.get_axis_name(i))
        for j in range(0, size):
            print(an.get_axis_name(j), end=': ')
            print(Analyzer.pearson_correlation(an.get_axis_data(i), an.get_axis_data(j)), end=' | ')
        print()


def correlation_graph(an: Analyzer):
    size = len(an.get_data()[0])
    for i in range(0, size):
        for j in range(0, size):
            x = an.get_axis_data(i)
            y = an.get_axis_data(j)
            if i != j:
                plt.plot(x, y, 'o')
                plt.title(an.get_axis_name(i) + ' and ' + an.get_axis_name(j))
            else:
                axis_data = an.get_axis_data(i)
                num_bins = Analyzer.bins_num(len(axis_data))
                plt.hist(axis_data, num_bins, density=True)
                plt.title(an.get_axis_name(i))
            plt.show()


def chekc_hypoth(an: Analyzer):
    size = len(an.get_data()[0])
    for i in range(0, size):
        hypothesis = Hypothesis()
        print(an.get_axis_name(i))
        hypothesis.check_hypothesis(an.get_axis_data(i))


def density(an: Analyzer):
    size = len(an.get_data()[0])
    colors = ['y', 'g', 'b']
    for i in range(0, 3):
        data = an.get_one(i)
        e_data = []
        for elem in data:
            e_data.append(elem[size - 1])
        empirical_density_function(e_data, colors[i])
    plt.show()


def main(argv: [str]):
    assert len(argv) > 0

    an = Analyzer(argv[0])
    # print(an.get_one(0))
    # print(an.get_data())
    # print(an.get_axis_data(1))
    # chars(an)
    # hist(an)
    # boxplot(an)
    # pca(an, 2)
    # correlation_matrix(an)
    # correlation_graph(an)
    # chekc_hypoth(an)
    # density(an)

    return


if __name__ == '__main__':
    main(sys.argv[1:])
