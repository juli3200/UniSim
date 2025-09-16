#include <iostream>
#include <cuda_runtime.h>

#include "helper.hpp"

#define u_long unsigned int
#define ThreadsPerBlock 256

template <typename T>
__global__ void clear_kernel(T* ptr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        ptr[i] = 0;
    }
}


template <typename T>
void clear_grid(T* grid, u_long size) {

    // define block sizes
    u_long blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

    // launch kernel
    clear_kernel<<<blockN, ThreadsPerBlock>>>(grid, size);
    // cudaDeviceSynchronize(); waiting here not necessary

}


extern "C"{
    // ----------------float memory management functions----------------

    // allocates memory on the GPU and returns a pointer to it
    float* alloc_f(u_long size){
        float* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(float));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_f(float* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_f(float* d_ptr, float* h_ptr, u_long size){
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_f(float* h_ptr, float* d_ptr, u_long size){
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_f(float* target, float* origin, u_long size){
        cudaMemcpy(target, origin, size, cudaMemcpyDeviceToDevice);
    }

    // clears memory on the device by setting all bytes to 0
    void clear_f(float* d_ptr, u_long size){
        clear_grid(d_ptr, size);
    }

    // ----------------u_long memory management functions----------------

    // allocates memory on the GPU and returns a pointer to it
    u_long* alloc_u(u_long size){
        u_long* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(u_long));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_u(u_long* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_u(u_long* d_ptr, u_long* h_ptr, u_long size){
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_u(u_long* h_ptr, u_long* d_ptr, u_long size){
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_u(u_long* target, u_long* origin, u_long size){
        cudaMemcpy(target, origin, size, cudaMemcpyDeviceToDevice);
    }

    // clears memory on the device by setting all bytes to 0
    void clear_u(u_long* d_ptr, u_long size){
        clear_grid(d_ptr, size);
    }

    // ----------------char memory management functions----------------
    // allocates memory on the GPU and returns a pointer to it
    char* alloc_c(u_long size){
        char* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(char));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_c(char* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_c(char* d_ptr, char* h_ptr, u_long size){
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_c(char* h_ptr, char* d_ptr, u_long size){
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_c(char* target, char* origin, u_long size){
        cudaMemcpy(target, origin, size, cudaMemcpyDeviceToDevice);
    }

    // clears memory on the device by setting all bytes to 0
    void clear_c(char* d_ptr, u_long size){
        clear_grid(d_ptr, size);
    }

}