from matplotlib import pyplot as plt
from generator import Generator
from regression import Regression


def f(x):
    return [2 + 2 * elem for elem in x]


def init():
    function = f
    section = [-1.8, 2]
    step_num = 20
    fluctuations = [10, -10]
    return function, section, step_num, fluctuations


def plot(x, y, func, params, name):
    plt.plot(x, y, 'o', label='Выборка')
    plt.plot(x, func(x), label='Модель')
    legends = ['МНК', 'МНМ']
    it = 0
    for param in params:
        assert len(param) == 2
        y1 = [param[0] * elem + param[1] for elem in x]
        plt.plot(x, y1, label=legends[it])
        it += 1
    plt.legend()
    plt.savefig('images/' + name + '.png')
    plt.show()


def calculate_regression(x, y, function, name):
    regression = Regression(x, y)
    a1, b1 = regression.least_squares_regression()
    print(a1, b1)
    a2, b2 = regression.least_absolute_regression()
    print(a2, b2)
    plot(x, y, function, [[a1, b1], [a2, b2]], name)


def main():
    function, section, step_num, fluctuations = init()
    generator = Generator(function, section)
    x1, y1 = generator.create_samples(step_num)
    x2, y2 = generator.create_sample_with_fluctuations(step_num, fluctuations)
    calculate_regression(x1, y1, function, 'WithoutFluctuation')
    calculate_regression(x2, y2, function, 'WithFluctuation')
    return 0


if __name__ == '__main__':
    main()
