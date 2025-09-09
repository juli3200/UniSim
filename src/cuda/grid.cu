#include <iostream>
#include <cuda_runtime.h>
#include <math_functions.h>

#define u_int unsigned int
#define ThreadsPerBlock 256


__global__ void fill_grid_kernel(u_int* grid, u_int* dim, u_int size, float* pos, u_int* cell, u_int* overflow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {

        u_int dim_x = dim[0];
        u_int depth = dim[2];

        // casting float positions to int for indexing
        // flooring is handled by the cast
        int x = (int)pos[i * 2];
        int y = (int)pos[i * 2 + 1];

        int index = (x + y * dim_x) * depth;
        int slot;

        // use this slower method to always fill the first available slot
        for (slot = 0; slot < depth; slot++) {
            // check if slot is occupied
            // and fill it if it's empty
            if (atomicCAS(&grid[index + slot], 0, i) == 0) {
                cell[i] = index + slot;
                break;
            }
        }

        if (slot == depth) {
            // if there's no space, increase the overflow counter
            // and do not add the id to the grid
            atomicAdd(overflow, 1);
        }

    }
}

__global__ void clear_grid_kernel(unsigned int* grid, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grid[i] = 0;
    }
}

extern "C" {
    
    // fills a 3D grid with a specified value
    // pointers are already device pointers
    int fill_grid(u_int size, u_int* dim, u_int* grid, float* pos, u_int* cell) {

        bool error = false;

        // allocate overflow counter
        u_int* d_overflow;
        cudaMalloc((void**)&d_overflow, sizeof(u_int));
        cudaMemset(d_overflow, 0, sizeof(u_int));

        // define block sizes
        u_int blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

        // launch kernel
        fill_grid_kernel<<<blockN, ThreadsPerBlock>>>(grid, dim, size, pos, cell, d_overflow);
        cudaError_t err = cudaDeviceSynchronize(); // wait for kernel to finish

        // check for launch errors
        if (err != cudaSuccess) {
            printf("Launch error: %s\n", cudaGetErrorString(err));
            error = true;
        }

        // copy overflow counter back to host and free device memory
        u_int h_overflow;
        cudaMemcpy(&h_overflow, d_overflow, sizeof(u_int), cudaMemcpyDeviceToHost);
        cudaFree(d_overflow);

        if (error) {
            return -1;
        }

        return h_overflow;
    }

    // clears a 3D grid by setting all values to 0
    void clear_grid(u_int* grid, u_int size) {

        // define block sizes
        u_int blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

        // launch kernel
        clear_grid_kernel<<<blockN, ThreadsPerBlock>>>(grid, size);
        cudaDeviceSynchronize(); // wait for kernel to finish

    }

}