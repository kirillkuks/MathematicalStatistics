import sys

from signal import Signal


def first_n_numbers(string: str, size: int):
    i = 0
    number_counter = 0
    while number_counter != size and i < len(string):
        i += 1
        if string[i] == ' ':
            number_counter += 1
    substring = string[:i]
    substring = substring.replace(',', '').replace('[', '')
    return substring.split(' ')


def read_signal(filename: str, size: int):
    with open(filename, 'r') as fin:
        line = fin.readline()
        signal_data = first_n_numbers(line, size)
    return [float(x) for x in signal_data]


def main(argv: []):
    if len(argv) == 2:
        signal_data = read_signal(argv[0], int(argv[1]))

        signal = Signal(signal_data)

        # signal.plot_signal_data()
        signal.plot_signal_hist()

        signal.apply_median_filter()
        # signal.plot_filtered_signal_data()

        signal.define_sections()
        signal.plot_data_signal_with_regions()

        signal.fisher_rule()
    return


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1:])
