from random import randint
import time
import csv
import logging


class Timer:

    def __init__(self):
        self._start_time = 0
        self._stop_time = 0

    def start(self):
        self._start_time = time.time()

    def stop(self):
        self._stop_time = time.time()

    def elapsed(self):
        return self._stop_time - self._start_time


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def generate_data(length, lower_range=0, upper_range=10000):
    """Generating random data"""

    data = []
    for _ in range(0, length):
        data.append(randint(lower_range, upper_range))
    return data


def export_to_csv(dict_array, name='result.csv'):
    """Generating csv file"""

    with open('result.csv', newline='', mode='w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict_array[0].keys())
        writer.writeheader()
        writer.writerows(dict_array)


def main():
    print(generate_data(20))


if __name__ == '__main__':
    main()
