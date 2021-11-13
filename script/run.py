"""Sorting algorithm using CPU."""

import argparse

from sorter import CPUSorter, GPUSorter
from util import export_to_csv, generate_data, init_logger

logger = init_logger('TEST_UTIL')


def test(sorter_cls, data_length):
    data = generate_data(data_length)
    sorter = sorter_cls(data)
    sorter.sort()
    return sorter.get_measured_times()


def run_test(sorter, data_func, iterations=5, attempts=10):
    return_info = []
    logger.info('Starting testing utility for %s...', sorter.__name__)
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
        average.update({'data_length': data_length})
        return_info.append(average)
    return return_info


def main(iterations, attempts, data_func):
    export_to_csv(
        run_test(GPUSorter, lambda x: eval(data_func), iterations, attempts),
        'gpu_result.csv')
    export_to_csv(
        run_test(CPUSorter, lambda x: eval(data_func), iterations, attempts),
        'cpu_result.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="CPU/GPU bubble sorting testing utility.")
    parser.add_argument('iterations', type=int)
    parser.add_argument('--attempts', dest='attempts', type=int, default=10)
    parser.add_argument('--data_func',
                        dest='data_func',
                        type=str,
                        default="100 * x**2")
    args = parser.parse_args()
    main(args.iterations, args.attempts, args.data_func)
