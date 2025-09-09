#include <iostream>
#include <cuda_runtime.h>

#include "src/cuda/cu_src/helper.hpp"

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

__device__ void border_collision(LigandArrays l_arrays, int i, u_int dim_x, u_int dim_y) {
    float x = l_arrays.pos_ligand[i * 2];
    float y = l_arrays.pos_ligand[i * 2 + 1];
    float vx = l_arrays.vel_ligand[i * 2];
    float vy = l_arrays.vel_ligand[i * 2 + 1];

    // check for border collisions and reflect velocity if necessary
    if (x < 0.0f) {
        l_arrays.pos_ligand[i * 2] = 0.0f;
        l_arrays.vel_ligand[i * 2] = -vx;
    } else if (x >= dim_x) {
        l_arrays.pos_ligand[i * 2] = dim_x;
        l_arrays.vel_ligand[i * 2] = -vx;
    }

    if (y < 0.0f) {
        l_arrays.pos_ligand[i * 2 + 1] = 0.0f;
        l_arrays.vel_ligand[i * 2 + 1] = -vy;
    } else if (y >= dim_y) {
        l_arrays.pos_ligand[i * 2 + 1] = dim_y;
        l_arrays.vel_ligand[i * 2 + 1] = -vy;
    }
}


__global__ void ligand_collision_kernel(u_int size, u_int search_radius, u_int* dim, u_int* grid, EntityArrays e_arrays, LigandArrays l_arrays, CollisionArrays col_arrays) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {

        if (l_arrays.message_ligand[i] == 0) {
            // skip if ligand is already collided
            return;
        }


        u_int dim_x = dim[0];
        u_int dim_y = dim[1];
        u_int depth = dim[2];


        // check for collisions with borders and reflect velocity if necessary
        border_collision(l_arrays, i, dim_x, dim_y);

        // casting float positions to int for indexing
        // flooring is handled by the cast
        float x = l_arrays.pos_ligand[i * 2];
        float y = l_arrays.pos_ligand[i * 2 + 1];

        // iterate over search area
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            // skip if out of bounds
            if ((int)x + dx < 0 || (int)x + dx >= (int)dim_x) {
                continue;
            }
            for (int dy = -search_radius; dy <= search_radius; dy ++){
                // skip if out of bounds
                if ((int)y + dy < 0 || (int)y + dy >= (int)dim_y) {
                    continue;
                }

                // compute grid index: (y * dim_x + x) * depth
                int index = (((int)y + dy) * dim_x + (int)x + dx) * depth;

                // check collisions in this cell
                // iterate over depth
                for (int slot = 0; slot < depth; slot++) {
                    u_int entity_index = grid[index + slot];
                    if (entity_index != 0) {
                        // compute distance
                        float dx = e_arrays.pos_entity[entity_index * 2] - x;
                        float dy = e_arrays.pos_entity[entity_index * 2 + 1] - y;
                        float dist_sq = dx * dx + dy * dy;

                        // check if collided
                        if (dist_sq <= search_radius * search_radius) {
                            // register collision
                            int old_val = atomicAdd(col_arrays.counter, 1);
                            col_arrays.collided_entities[old_val] = e_arrays.id_entity[entity_index];
                            col_arrays.collided_pos[old_val * 2] = x;
                            col_arrays.collided_pos[old_val * 2 + 1] = y;
                            col_arrays.collided_message[old_val] = l_arrays.message_ligand[i];

                            // delete ligand by setting its message to 0
                            l_arrays.message_ligand[i] = 0;

                        }
                    }
                }

            }
        }




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

    // performs collision detection for ligands against entities in a grid
    // pointers are already device pointers
    


}