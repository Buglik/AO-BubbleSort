from typing import Dict, List

import numpy
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import DeviceAllocation
import pycuda.driver as cuda

from util import Timer


class Sorter:
    def __init__(self, array) -> None:
        self.original_array = numpy.array(array).astype(numpy.int32)
        self._timer = Timer()
        self._sorting_time = 0

    def sort(self) -> List:
        pass

    def get_measured_times(self) -> Dict:
        return {"sort_time": self._sorting_time}


class CPUSorter(Sorter):
    def __init__(self, array) -> None:
        super().__init__(array)
        self._sorting_time = 0

    def sort(self) -> List:
        tmp = self.original_array.copy()
        self._timer.start()
        for last_index in range(len(tmp) - 1, -1, -1):
            for curr_index in range(last_index):
                if tmp[curr_index] > tmp[curr_index + 1]:
                    tmp[curr_index], tmp[curr_index +
                                         1] = tmp[curr_index +
                                                  1], tmp[curr_index]
        self._timer.stop()
        self._sorting_time = self._timer.elapsed()
        return tmp


class GPUSorter(Sorter):

    ODDEVEN_KERNEL = """

        __global__ void bubbleSortOdd(int *inputArray, int size) {
            
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

        __global__ void bubbleSortEven(int *inputArray, int size) {
            
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
        """

    OLD_KERNEL = """
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
    """

    def __init__(self, array) -> None:
        super().__init__(array)
        self._copying_time = 0
        self._sorting_time = 0
        self._receiving_time = 0

    def sort(self):
        gpu_array = self._allocate_and_copy()
        self._sort_odd_even(gpu_array)

        return self._retrieve_array(gpu_array)

    def _retrieve_array(self, gpu_array):
        sorted_array = numpy.empty_like(self.original_array)
        self._timer.start()
        cuda.memcpy_dtoh(sorted_array, gpu_array)
        self._timer.stop()
        self._receiving_time = self._timer.elapsed()
        return sorted_array

    def _sort_odd_even(self, gpu_array) -> None:

        entry_arr_size = numpy.uint32(self.original_array.size)

        block, grid = self._get_gpu_params()

        mod = SourceModule(GPUSorter.ODDEVEN_KERNEL)
        odd_sort = mod.get_function('bubbleSortOdd')
        even_sort = mod.get_function('bubbleSortEven')
        self._timer.start()
        for _ in range(entry_arr_size):
            odd_sort(gpu_array, entry_arr_size, block=block, grid=grid)
            even_sort(gpu_array, entry_arr_size, block=block, grid=grid)
        self._timer.stop()
        self._sorting_time = self._timer.elapsed()

    def _get_gpu_params(self):
        # if array is larger than max threads,
        # then we need to create larger grid with max size (1024)
        block_size = self.original_array.size
        grid = (1, 1, 1)

        if self.original_array.size > 1024:
            block_size = 1024
            grid = (int(self.original_array.size / block_size) + 1, 1, 1)
        block = (block_size, 1, 1)

        return block, grid

    def _allocate_and_copy(self) -> DeviceAllocation:
        gpu_array = cuda.mem_alloc(self.original_array.nbytes)
        self._timer.start()
        cuda.memcpy_htod(gpu_array, self.original_array)
        self._timer.stop()
        self._copying_time = self._timer.elapsed()
        return gpu_array

    def get_measured_times(self) -> Dict:
        return {
            "copy_time": self._copying_time,
            "sort_time": self._sorting_time,
            "retrieve_time": self._receiving_time
        }
