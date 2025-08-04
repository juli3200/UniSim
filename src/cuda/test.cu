#include <iostream>
#include <cuda_runtime.h>

extern "C"{
    int main(){
        cudaError_t err = cudaSuccess;

        // Check if CUDA is available
        int deviceCount;
        err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            std::cerr << "Error getting device count: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        if (deviceCount == 0) {
            std::cout << "No CUDA devices available." << std::endl;
            return -1;
        }

        std::cout << "CUDA devices available: " << deviceCount << std::endl;

        return 0;
    }
}