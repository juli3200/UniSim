#include <iostream>
#include <cuda_runtime.h>

extern "C"{
    int* alloc_memory_cu(int i){
        int* d_ptr;
        cudaMalloc((void**)&d_ptr, sizeof(int));
        cudaMemcpy(d_ptr, &i, sizeof(int), cudaMemcpyHostToDevice);

        return d_ptr;
    }

    int release_memory_cu(int* d_ptr){
        int* h_ptr = new int;
        cudaMemcpy(h_ptr, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_ptr);
        return *h_ptr;
    }



    int cuda_test(){
        cudaError_t err = cudaSuccess;

        // Check if CUDA is available
        int deviceCount;
        err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess) {
            return -1;
        }

        if (deviceCount == 0) {
            return 2; // No CUDA devices available
        }


        printf("CUDA device count: %d\n", deviceCount);
        printf("CUDA device properties:\n");
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            err = cudaGetDeviceProperties(&deviceProp, i);
            if (err != cudaSuccess) {
                return -1;
            }
            printf("Device %d: %s\n", i, deviceProp.name);
            printf("  Total Global Memory: %zu bytes\n", deviceProp.totalGlobalMem);

        }

        return 0;
    }
}