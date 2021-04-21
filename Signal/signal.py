import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


class SectionType(Enum):
    background = (0, 'y')
    transition = (1, 'g')
    signal = (2, 'r')

    def __init__(self, id, title):
        self.id = id
        self.title = title


def define_color(section_type: SectionType):
    if section_type.id == 0:
        return 'фон'
    elif section_type.id == 1:
        return 'переход'
    else:
        return 'сигнал'


class Section:
    def __init__(self, left: int, right: int, section_type: SectionType):
        self.left = left
        self.right = right
        self.section_type = section_type

    def get_edges(self):
        return self.left, self.right

    def get_linspace(self):
        return np.linspace(self.left, self.right, self.right - self.left + 1)

    def get_section_type(self):
        return self.section_type


class Signal:
    def __init__(self, signal_data: []):
        self.signal_data = signal_data
        self.size = len(signal_data)
        self.filtered_signal = []
        self.hist_data = []
        self.sections = []

    def _plot(self, signal_data):
        x = np.linspace(0, self.size - 1, self.size)
        plt.plot(x, signal_data)

    def plot_signal_data(self):
        self._plot(self.signal_data)
        plt.savefig('images/SignalData.png')
        plt.show()

    def plot_filtered_signal_data(self):
        assert self.filtered_signal
        self._plot(self.filtered_signal)
        plt.savefig('images/Filtered.png')
        plt.show()

    def plot_signal_hist(self):
        number_of_intervals = int(1.72 * (self.size ** (1 / 3)))
        self.hist_data = plt.hist(self.signal_data, bins=number_of_intervals)
        plt.savefig('images/Histogram.png')
        plt.show()

    def apply_median_filter(self):
        self.filtered_signal = self.signal_data
        self.filtered_signal = self.signal_data
        for i in range(1, self.size - 1):
            mean = (self.signal_data[i - 1] + self.signal_data[i + 1]) / 2
            if mean < self.filtered_signal[i]:
                self.filtered_signal[i] = mean

    def define_sections(self):
        assert self.hist_data
        bins = [int(x) for x in self.hist_data[0]]

        sections = [float(x) for x in self.hist_data[1]]
        background_section_index = bins.index(max(bins))
        signal_section_index = bins.index(max(bins[:background_section_index] + bins[background_section_index + 1:]))

        signal_values = [sections[signal_section_index], sections[signal_section_index + 1]]
        background_values = [sections[background_section_index], sections[background_section_index + 1]]

        self.sections = self._define_sections_types(signal_values, background_values)

    def _define_sections_types(self, signal_values: [], backgrounds_values: []):
        assert self.filtered_signal
        assert len(signal_values) == len(backgrounds_values) == 2

        signal_types = [SectionType.transition for _ in range(self.size)]

        for i in range(self.size):
            if signal_values[0] <= self.filtered_signal[i] <= signal_values[1]:
                signal_types[i] = SectionType.signal
            if backgrounds_values[0] <= self.filtered_signal[i] <= backgrounds_values[1]:
                signal_types[i] = SectionType.background

        sections = []
        cur_type = signal_types[0]
        last_edge = 0
        for i in range(self.size):
            if cur_type.id != signal_types[i].id:
                sections.append(Section(last_edge, i - 1, cur_type))
                last_edge = i
                cur_type = signal_types[i]
        sections.append(Section(last_edge, self.size - 1, cur_type))

        return sections

    def plot_data_signal_with_regions(self):
        assert self.sections
        assert self.filtered_signal
        for section in self.sections:
            left, right = section.get_edges()
            section_type = section.get_section_type()
            plt.plot(section.get_linspace(), self.filtered_signal[left:right + 1],
                     color=section_type.title, label=define_color(section_type))
        plt.legend()
        plt.savefig('images/Regions.png')
        plt.show()

    @staticmethod
    def number_of_splitting(size: int):
        return int(1.72 * (size ** (1 / 3)))

    @staticmethod
    def _intar_group(sample: [], k: int):
        size = len(sample)
        delta = int(size / k)
        delta1 = int(delta + size % k)

        inter = 0
        inter += Signal.variance(sample[:delta1])

        left = delta1
        for i in range(1, k):
            inter += Signal.variance(sample[left:left + delta])
            left += delta

        return inter / k

    @staticmethod
    def _inter_group(sample: [], k: int):
        size = len(sample)
        delta = int(size / k)
        delta1 = int(delta + size % k)

        means = [Signal.mean(sample[:delta1])]

        left = delta1
        for i in range(1, k):
            means.append(Signal.mean(sample[left:left + delta]))
            left += delta

        return Signal.variance(means) * k

    def _fisher(self, left, right):
        sample = self.filtered_signal[left:right + 1]
        size = right - left + 1
        k = Signal.number_of_splitting(size)
        intar = Signal._intar_group(sample, k)
        inter = Signal._inter_group(sample, k)
        return inter / intar

    def fisher_rule(self):
        assert self.sections
        assert self.filtered_signal
        fisher_coefficients = []
        for section in self.sections:
            left, right = section.get_edges()
            fisher_coefficients.append(self._fisher(left, right))
        print(fisher_coefficients)
        return

    @staticmethod
    def mean(sample):
        s = 0
        for elem in sample:
            s += elem
        return s / len(sample)

    @staticmethod
    def variance(sample):
        mean = Signal.mean(sample)
        s = 0
        for elem in sample:
            s += (elem - mean) ** 2
        return s / (len(sample) - 1)

