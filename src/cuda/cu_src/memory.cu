#include <iostream>
#include <cuda_runtime.h>
#include <string>

#define u_int unsigned int
#define ThreadsPerBlock 256

template <typename T>
__global__ void clear_kernel(T* ptr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        ptr[i] = 0;
    }
}


template <typename T>
void clear_grid(T* grid, u_int size) {

    // define block sizes
    u_int blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

    // launch kernel
    clear_kernel<<<blockN, ThreadsPerBlock>>>(grid, size);
    // cudaDeviceSynchronize(); waiting here not necessary

}


extern "C"{
    // ----------------float memory management functions----------------

    // allocates memory on the GPU and returns a pointer to it
    float* alloc_f(u_int size){
        float* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(float));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_f(float* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_f(float* d_ptr, float* h_ptr, u_int size){
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_f(float* h_ptr, float* d_ptr, u_int size){
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_f(float* target, float* origin, u_int size){
        cudaMemcpy(target, origin, size, cudaMemcpyDeviceToDevice);
    }

    // clears memory on the device by setting all bytes to 0
    void clear_f(float* d_ptr, u_int size){
        clear_grid(d_ptr, size);
    }

    // ----------------u_int memory management functions----------------

    // allocates memory on the GPU and returns a pointer to it
    u_int* alloc_u(u_int size){
        u_int* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(u_int));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_u(u_int* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_u(u_int* d_ptr, u_int* h_ptr, u_int size){
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_u(u_int* h_ptr, u_int* d_ptr, u_int size){
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_u(u_int* target, u_int* origin, u_int size){
        cudaMemcpy(target, origin, size, cudaMemcpyDeviceToDevice);
    }

    // clears memory on the device by setting all bytes to 0
    void clear_u(u_int* d_ptr, u_int size){
        clear_grid(d_ptr, size);
    }

    // ----------------char memory management functions----------------
    // allocates memory on the GPU and returns a pointer to it
    char* alloc_c(u_int size){
        char* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(char));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_c(char* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_c(char* d_ptr, char* h_ptr, u_int size){
        cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_c(char* h_ptr, char* d_ptr, u_int size){
        cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_c(char* target, char* origin, u_int size){
        cudaMemcpy(target, origin, size, cudaMemcpyDeviceToDevice);
    }

    // clears memory on the device by setting all bytes to 0
    void clear_c(char* d_ptr, u_int size){
        clear_grid(d_ptr, size);
    }

}