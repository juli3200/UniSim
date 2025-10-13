#include <iostream>
#include <cuda_runtime.h>

#include "helper.hpp"

#define ThreadsPerBlock 256
#define COLLISION_SPACE_FACTOR 0.2 // factor to allocate more space for collided ligands


__global__ void fill_grid_kernel(uint16_t* grid, Dim dim, uint16_t size, EntityCuda* entities, uint32_t* overflow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {

        uint32_t dim_x = dim.x;
        uint32_t depth = dim.depth;

        // casting float positions to int for indexing
        // flooring is handled by the cast
        int x = (int)entities[i].posx;
        int y = (int)entities[i].posy;


        int index = (x + y * dim_x) * depth;
        int slot;

        // use this slower method to always fill the first available slot
        for (slot = 0; slot < depth; slot++) {
            // check if slot is occupied
            // and fill it if it's empty

            // CAS Compare And Swap
            if (atomicCAS(&grid[index + slot], 0, i) == 0) {
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

// simple kernel to update ligand positions based on their velocities
__global__ void update_positions_kernel(LigandArrays l_arrays, float delta_time) {

int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < l_arrays.num_ligands) {
        l_arrays.pos[i * 2] += l_arrays.vel[i * 2] * delta_time;
        l_arrays.pos[i * 2 + 1] += l_arrays.vel[i * 2 + 1] * delta_time;
        
    }
}

__device__ void border_collision(LigandArrays l_arrays, int i, uint32_t dim_x, uint32_t dim_y) {
    float x = l_arrays.pos[i * 2];
    float y = l_arrays.pos[i * 2 + 1];
    float vx = l_arrays.vel[i * 2];
    float vy = l_arrays.vel[i * 2 + 1];

    // check for border collisions and reflect velocity if necessary
    if (x < 0.0f) {
        l_arrays.pos[i * 2] = 0.0f;
        l_arrays.vel[i * 2] = -vx;
    } else if (x >= dim_x) {
        l_arrays.pos[i * 2] = dim_x;
        l_arrays.vel[i * 2] = -vx;
    }

    if (y < 0.0f) {
        l_arrays.pos[i * 2 + 1] = 0.0f;
        l_arrays.vel[i * 2 + 1] = -vy;
    } else if (y >= dim_y) {
        l_arrays.pos[i * 2 + 1] = dim_y;
        l_arrays.vel[i * 2 + 1] = -vy;
    }
}

__global__ void ligand_collision_kernel(uint32_t size, uint32_t search_radius, Dim dim, uint32_t* grid, EntityCuda* entities, LigandCuda* ligands, LigandCuda* collided_ligands, uint32_t* counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {

        if (ligands[i].message == 0) {
            // skip if ligand is already collided
            return;
        }


        uint32_t dim_x = dim.x;
        uint32_t dim_y = dim.y;
        uint32_t depth = dim.depth;


        // check for collisions with borders and reflect velocity if necessary
        border_collision(l_arrays, i, dim_x, dim_y);

        // casting float positions to int for indexing
        // flooring is handled by the cast
        float x = l_arrays.pos[i * 2];
        float y = l_arrays.pos[i * 2 + 1];

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
                    uint32_t entity_index = grid[index + slot];
                    if (entity_index != 0) {
                        // compute distance
                        float dx = e_arrays.pos[entity_index * 2] - x;
                        float dy = e_arrays.pos[entity_index * 2 + 1] - y;
                        float dist_sq = dx * dx + dy * dy;

                        // check if collided
                        if (dist_sq <= search_radius * search_radius) {
                            // register collision
                            int old_val = atomicAdd(col_arrays.counter, 1);
                            col_arrays.collided_entities[old_val] = e_arrays.id[entity_index];
                            col_arrays.collided_pos[old_val * 2] = x;
                            col_arrays.collided_pos[old_val * 2 + 1] = y;
                            col_arrays.collided_message[old_val] = l_arrays.message[i];

                            // delete ligand by setting its message to 0
                            l_arrays.message[i] = 0;

                        }
                    }
                }

            }
        }




    }
}

LigandWrapper error_return(LigandCuda* collided_ligands, uint32_t* counter) {
    // free device memory
    cudaFree(collided_ligands);
    cudaFree(counter);

    LigandWrapper empty_wrapper;
    empty_wrapper.ligands = nullptr;
    empty_wrapper.count = 0;
    return empty_wrapper;
}



extern "C" {
    
    // fills a 3D grid with a specified value
    // pointers are already device pointers
    int fill_grid(uint32_t size, Dim dim, uint16_t* grid, EntityCuda* entities) {

        bool error = false;
        printf("Size: %lu\n", size);
        printf("Dim: %lu x %lu x %lu\n", dim.x, dim.y, dim.depth);

        // allocate overflow counter
        uint32_t* d_overflow;
        cudaMalloc((void**)&d_overflow, sizeof(uint32_t));
        cudaMemset(d_overflow, 0, sizeof(uint32_t));

        // define block sizes
        uint32_t blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

        // launch kernel
        fill_grid_kernel<<<blockN, ThreadsPerBlock>>>(grid, dim, size, entities, d_overflow);
        cudaError_t err = cudaDeviceSynchronize(); // wait for kernel to finish

        // check for launch errors
        if (err != cudaSuccess) {
            printf("Launch error: %s\n", cudaGetErrorString(err));
            error = true;
        }

        // copy overflow counter back to host and free device memory
        uint32_t h_overflow;
        cudaMemcpy(&h_overflow, d_overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_overflow);

        printf("Overflow: %lu\n", h_overflow);

        if (error) {
            return -1;
        }

        return h_overflow;
    }

    // performs collision detection for ligands against entities in a grid
    // pointers are already device pointers
    LigandWrapper ligand_collision(uint32_t search_radius, Dim dim, uint32_t* grid, EntityCuda* entities, LigandCuda* ligands, uint32_t ligands_size) {

        LigandCuda* collided_ligands = ligands;
        cudaMemset(collided_ligands, 0, ligands_size * sizeof(LigandCuda) * COLLISION_SPACE_FACTOR); // clear collided ligands array

        uint32_t* counter;
        cudaMemset(&counter, sizeof(uint32_t), 0); // initialize counter to 0


        // launch kernel
        uint32_t blockN = (ligands_size + ThreadsPerBlock - 1) / ThreadsPerBlock;
        ligand_collision_kernel<<<blockN, ThreadsPerBlock>>>(ligands_size, search_radius, dim, grid, entities, ligands, collided_ligands, counter);
        cudaError_t err = cudaDeviceSynchronize(); // wait for kernel to finish

        // check for launch errors
        if (err != cudaSuccess) {
            printf("ligand_collision failed\n");
            printf("Launch error: %s\n", cudaGetErrorString(err));
            return error_return(collided_ligands, counter);
        }

        // copy counter back to host
        uint32_t h_counter;
        cudaMemcpy(&h_counter, counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (h_counter > ligands_size * COLLISION_SPACE_FACTOR) {
            // overflow, too many collisions
            printf("Error: counter exceeds allocated size\n");
            return error_return(collided_ligands, counter);
        }



        // free device memory
        cudaFree(counter);
        cudaFree(collided_ligands);

        // prepare return struct
        LigandWrapper h_collided;
        h_collided.ligands = collided_ligands;
        h_collided.count = h_counter;

        return h_collided;

    }

    // updates ligand positions based on their velocities
    void update_positions(LigandArrays l_arrays, float delta_time) {
        int size = l_arrays.num_ligands;
        uint32_t blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

        // launch kernel
        update_positions_kernel<<<blockN, ThreadsPerBlock>>>(l_arrays, delta_time);
        cudaError_t err = cudaDeviceSynchronize(); // wait for kernel to finish

        // check for launch errors
        if (err != cudaSuccess) {
            printf("Launch error: %s\n", cudaGetErrorString(err));
        }
    }

}


