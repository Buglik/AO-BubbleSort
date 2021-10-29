"""Sorting algorithm using CPU."""

from util import generate_data, Timer, export_to_csv, init_logger

logger = init_logger('TEST_UTIL')

def cpu_bubble_sort(array):
    tmp = array.copy()
    for last_index in range(len(tmp)-1, -1, -1):
        for curr_index in range(last_index):
            if tmp[curr_index] > tmp[curr_index+1]:
                tmp[curr_index], tmp[curr_index+1] = tmp[curr_index+1], tmp[curr_index]
    return tmp

def test(test_func, data_length):
    timer = Timer()
    data = generate_data(data_length)
    timer.start()
    test_func(data)
    timer.stop()
    return timer.elapsed()

def run_test(test_func, data_func, iterations=5, attempts=10):
    return_info = []
    logger.info('Starting testing utility for provided function...')
    logger.info(f'ITERATIONS: {iterations}')
    logger.info(f'ATTEMPTS (TO GET AVERAGE): {attempts}')
    for length in range(1, iterations+1):
        data_length = data_func(length)
        average = 0
        for attempt in range(attempts):
            average += test(test_func, data_length)
        average /= attempts
        logger.info(f'Length: {data_length} | Elapsed (average): {average} s')
        return_info.append({'data_length': data_length, 'elapsed_time': average})
    return return_info

def main():
    export_to_csv(run_test(cpu_bubble_sort, lambda x: 100*x, 20))

if __name__ == '__main__':
    main()