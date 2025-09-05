#include <iostream>
#include <cuda_runtime.h>

extern "C"{
    // allocates memory on the GPU and returns a pointer to it
    float* alloc_float(float* i, int size){
        float* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(float));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_float(float* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD(float* d_ptr, float* h_ptr, int size){
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH(float* h_ptr, float* d_ptr, int size){
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    }
}