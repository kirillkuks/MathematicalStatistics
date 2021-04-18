from confidence_interval import ConfidenceInterval
from normal_distribution import NormalDistribution


def init():
    sizes = [20, 100]
    confidence_level = 0.95
    return sizes, confidence_level


def main():
    sizes, confidence_level = init()
    normal = NormalDistribution(0, 1)

    for size in sizes:
        print('Size = ' + str(size))
        sample = normal.create_sample(size)
        confidence_interval = ConfidenceInterval(sample, confidence_level)
        left, right = confidence_interval.mean_confidence_interval_for_normal()
        print('Mean:')
        print(left, right)
        left, right = confidence_interval.deviation_confidence_interval_for_normal()
        print('Deviation:')
        print(left, right)
        left, right = confidence_interval.mean_asymptotic_confidence_interval()
        print('Asymptotic Mean:')
        print(left, right)
        left, right = confidence_interval.deviation_asymptotic_confidence_interval()
        print('Asymptotic Deviation:')
        print(left, right)


if __name__ == '__main__':
    main()
