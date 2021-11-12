"""Sorting algorithm using CPU."""

from util import generate_data, Timer, export_to_csv, init_logger
import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

logger = init_logger('TEST_UTIL')


def cpu_bubble_sort(array):
    tmp = array.copy()
    for last_index in range(len(tmp) - 1, -1, -1):
        for curr_index in range(last_index):
            if tmp[curr_index] > tmp[curr_index + 1]:
                tmp[curr_index], tmp[curr_index + 1] = tmp[curr_index + 1], tmp[curr_index]
    return tmp

def gpu_bubble_sort(array):

    float_arr = numpy.array(array).astype(numpy.float32)
    gpu_array = cuda.mem_alloc(float_arr.nbytes)
    cuda.memcpy_htod(gpu_array, float_arr)

    BLOCK_SIZE = float_arr.size
    grid = (1, 1, 1)
    if float_arr.size > 1024:
        BLOCK_SIZE = 1024
        grid = (int(float_arr.size/BLOCK_SIZE) + 1, 1, 1)
    block = (BLOCK_SIZE, 1, 1)
    
    mod = SourceModule("""
    
    __global__ void bubbleSort(float *inputArray, int size) {

        int idx = threadIdx.x;
        int N = size-1;
        
        for(int i = idx; i <= N; i++) {
            for(int j = 0; j <= N-1-i; j++) {
                if(inputArray[j] > inputArray[j+1]) {
                    float tmp = inputArray[j];
                    inputArray[j] = inputArray[j+1];
                    inputArray[j+1] = tmp;
                }
            }
        }
    }

    __global__ void bubbleSortOdd(float *inputArray, int size) {
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i % 2 == 0 && i < size-1){
            if(inputArray[i+1] < inputArray[i]){
                // switch in the x array
                int temp = inputArray[i];
                inputArray[i] = inputArray[i+1];
                inputArray[i+1] = temp;
            }
        }
    }

    __global__ void bubbleSortEven(float *inputArray, int size) {
        
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i % 2 != 0 && i < size-1){
            if(inputArray[i+1] < inputArray[i]){
                // switch in the x array
                int temp = inputArray[i];
                inputArray[i] = inputArray[i+1];
                inputArray[i+1] = temp;
            }
        }
    }
    """)
    func = mod.get_function('bubbleSortOdd')
    func2 = mod.get_function('bubbleSortEven')
    for _ in range(float_arr.size):
        func(gpu_array, numpy.int32(float_arr.size), block=block, grid=grid)
        func2(gpu_array, numpy.int32(float_arr.size), block=block, grid=grid)

    sorted_array = numpy.empty_like(float_arr)
    cuda.memcpy_dtoh(sorted_array, gpu_array)

    return sorted_array.astype(numpy.int64)






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
    for length in range(1, iterations + 1):
        data_length = data_func(length)
        average = 0
        for attempt in range(attempts):
            average += test(test_func, data_length)
        average /= attempts
        logger.info(f'Length: {data_length} | Elapsed (average): {average} s')
        return_info.append({'data_length': data_length, 'elapsed_time': average})
    return return_info


def main():
    # export_to_csv(run_test(cpu_bubble_sort, lambda x: 100 * x, 20))
    run_test(cpu_bubble_sort, lambda x: 100*x**2, 100, 10)
    # arr = generate_data(10000)
    # sorted_gpu = gpu_bubble_sort(arr)
    # sorted_cpu = numpy.array(cpu_bubble_sort(arr))
    # while(numpy.array_equal(sorted_gpu, sorted_cpu)):
    #     arr = generate_data(10000)
    #     sorted_gpu = gpu_bubble_sort(arr)
    #     sorted_cpu = numpy.array(cpu_bubble_sort(arr))
    #     print('jebac policje')
    # print(sorted_gpu)
    # print(sorted_cpu)


if __name__ == '__main__':
    main()
