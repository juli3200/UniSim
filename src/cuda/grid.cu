#include <iostream>
#include <cuda_runtime.h>
#include <math_functions.h>

__global__ void fill_grid_kernel(unsigned int* grid, unsigned int* dim, float* posx, float* posy, unsigned int* id, int size, int* cell, int* overflow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {

        unsigned int dim_x = dim[0];
        unsigned int dim_y = dim[1];
        unsigned int depth = dim[2];

        // casting float positions to int for indexing
        // flooring is handled by the cast
        int x = (int)posx[i];
        int y = (int)posy[i];
        unsigned int id_value = id[i];

        int index = (x + y * dim_x) * depth;

        // access the cell coord to see how many items are already there
        int slot = atomicAdd(&cell[x + y * dim_x], 1);

        // if there's space in the cell, add the id to the grid
        if (slot < depth) {
            grid[index + slot] = id_value;
        } else {
            // if there's no space, increase the overflow counter
            // and do not add the id to the grid
            atomicAdd(overflow, 1);
        }

    }
}

extern "C" {
    
    // fills a 3D grid with a specified value
    int fill_grid(unsigned int* grid, int* dim, float* posx, float* posy, unsigned int* id) {
        int size = dim[0] * dim[1] * dim[2];
        
        return 0;
    }

}