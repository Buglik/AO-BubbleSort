"""Sorting algorithm using CPU."""

from util import generate_data, export_to_csv, init_logger
from sorter import CPUSorter, GPUSorter

logger = init_logger('TEST_UTIL')


def test(sorter_cls, data_length):
    data = generate_data(data_length)
    sorter = sorter_cls(data)
    sorter.sort()
    return sorter.get_measured_times()


def run_test(sorter, data_func, iterations=5, attempts=10):
    return_info = []
    logger.info('Starting testing utility for provided sorter...')
    logger.info('ITERATIONS: %i', iterations)
    logger.info('ATTEMPTS (TO GET AVERAGE): %i', attempts)
    for length in range(1, iterations + 1):
        data_length = data_func(length)
        average = test(sorter, data_length)
        for _ in range(attempts - 1):
            results = test(sorter, data_length)
            for key in results.keys():
                average[key] += results[key]
        average = {k: v / attempts for k, v in average.items()}

        logger.info('Length: %i | Averages: %s', data_length, average)
        return_info.append({'data_length': data_length})
        return_info.append(average)
    return return_info


def main():
    run_test(GPUSorter, lambda x: 100 * x**2, 10, 10)
    # array = generate_data(10000)
    # gpu_sorter = GPUSorter(array)
    # gpu_sorter.sort()
    # print(gpu_sorter.get_measured_times())


if __name__ == '__main__':
    main()
