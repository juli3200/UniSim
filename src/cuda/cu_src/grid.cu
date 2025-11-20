#include <iostream>
#include <cuda_runtime.h>

#include "helper.hpp"

// 0xFFFFFFFF indicates empty either in grid or collided ligand

#define ThreadsPerBlock 256
const float COLLISION_SPACE_FACTOR = 0.5f; // factor to allocate more space for collided ligands
__device__ __constant__ float reciptocal_pi = 0.31830988618f; // 1/pi
__device__ __constant__ float rec_two_pow_32 = 0.00000000023283064365386962890625f; // 1/2^32

__global__ void fill_grid_kernel(uint32_t* grid, Dim dim, uint32_t size, EntityCuda* entities, uint32_t* overflow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {

        uint32_t dim_x = dim.x;
        uint32_t depth = dim.depth;

        // casting float positions to int for indexing
        // flooring is handled by the cast
        int x = (int)entities[i].posx;
        int y = (int)entities[i].posy;

        // check bounds
        if (x < 0 || x >= (int)dim_x || y < 0 || y >= (int)dim.y) {
            // out of bounds, do not add to grid
            return;
        }

        int index = (x + y * dim_x) * depth;
        int slot;

        // use this slower method to always fill the first available slot
        for (slot = 0; slot < depth; slot++) {
            // check if slot is occupied
            // and fill it if it's empty

            // CAS Compare And Swap
            if (atomicCAS(&grid[index + slot], 0xFFFFFFFF, i) == 0xFFFFFFFF) {
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
__global__ void update_positions_kernel(LigandCuda* ligands, uint32_t size, float delta_time) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (ligands[i].emitted_id == 0xFFFFFFFF) {
            // skip if ligand is already collided
            return;
        }
        ligands[i].posx += ligands[i].velx * delta_time;
        ligands[i].posy += ligands[i].vely * delta_time;
        
    }
}

__device__ bool border_collision(LigandCuda* ligands, int i, uint32_t dim_x, uint32_t dim_y) {
    float x = ligands[i].posx;
    float y = ligands[i].posy;
    float vx = ligands[i].velx;
    float vy = ligands[i].vely;

    // check for border collisions and reflect velocity if necessary
    if (x < 0.0f) {
        ligands[i].posx = 0.0f;
        ligands[i].velx = -vx;
        return true;
    } else if (x >= dim_x) {
        ligands[i].posx = dim_x;
        ligands[i].velx = -vx;
        return true;
    } else if (y < 0.0f) {
        ligands[i].posy = 0.0f;
        ligands[i].vely = -vy;
        return true;
    } else if (y >= dim_y) {
        ligands[i].posy = dim_y;
        ligands[i].vely = -vy;
        return true;
    }
    return false;
}

// https://github.com/skeeto/hash-prospector
__device__ float pseudo_random(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;

    return (float)x * rec_two_pow_32; // map to [0, 1)
    
}

__device__ void reflect_ligand(int index, CollisionUtils col_arrays, uint32_t entity_index, float dx, float dy, float velx, float vely) {
    // reflect ligand and continue

    // reflect on circle -> v- 2*(v*n)*n where n is the normal
    // n = (dx, dy)/size 
    // v = (velx, vely)


    float size_recip = 1.0f / col_arrays.entities[entity_index].size;

    float normal_x = dx * size_recip;
    float normal_y = dy * size_recip;

    float v_dot_n = velx * normal_x + vely * normal_y;

    col_arrays.ligands[index].velx = velx - 2.0f * v_dot_n * normal_x;
    col_arrays.ligands[index].vely = vely - 2.0f * v_dot_n * normal_y;
}

// check if ligand can bind to entity based on specs and angle of incidence
__device__ bool entity_collision(int index, CollisionUtils col_arrays, uint32_t entity_index, uint32_t n_receptors, float x, float y) {
    // return: collission occured

    // compute distance
    float dx = col_arrays.entities[entity_index].posx - x;
    float dy = col_arrays.entities[entity_index].posy - y;
    float dist_sq = dx * dx + dy * dy;

    float size_sq = col_arrays.entities[entity_index].size * col_arrays.entities[entity_index].size;

    // check if collided
    if (dist_sq > size_sq) {
        // no collision
        return false;
    }

    // check if ligand can bind to entity
    uint32_t ligand_spec = col_arrays.ligands[index].spec;
    
    // where the receptor specs for this entity start
    uint32_t receptor_start_index = col_arrays.entities[entity_index].receptor_id * n_receptors;

    float velx = col_arrays.entities[entity_index].velx;
    float vely = col_arrays.entities[entity_index].vely;

    // compute angle between ligand velocity and vector from ligand to entity center

    // maybe optimize division later
    float dot = dx * velx + dy * vely;
    // rsqrtf returns 1/sqrt(x)
    // instead of rsqrt(v1...) * rsqrt(v2...) we can do rsqrtf(v1... * v2...)
    float denominator = rsqrtf(
        (dist_sq) * (velx * velx + vely * vely)
    );
    
    float cos_angle = dot * denominator;

    float angle = acosf(cos_angle);

    if (isnan(angle)) {
        // invalid angle, send back where it came from
        // cannot reflect because vel is 0
        col_arrays.ligands[index].velx = -velx;
        col_arrays.ligands[index].vely = -vely;
        return true;
    }

    int relative_index = (int)(angle * reciptocal_pi * (float)n_receptors); // map angle to [0, n_receptors)
    // clamp to valid range
    if (relative_index >= (int)n_receptors) {
        relative_index = n_receptors - 1;
    }

    int receptor_index = receptor_start_index + relative_index;

    // check if there is a receptor at this index
    if (col_arrays.receptors[receptor_index] == 0) {
        // no receptor, reflect ligand and continue
        reflect_ligand(index, col_arrays, entity_index, dx, dy, velx, vely);
        return true;
    }

    /* old method to
    // match specs
    // first 16 bits are ignored because they are always 0
    int ones = __popc((uint32_t)(ligand_spec ^ col_arrays.receptors[receptor_index]));
    int matches = 16 - ones; // number of matching bits
    float match_prob = (float)matches / 16.0f;
    // generate pseudo-random number with unique seed between 0 and 1
    // seed = (x + y - dy) * 50000 + entity_index + index (just some values to mix it up)
    float rand = pseudo_random((int)((x + y - dy) * 50000) + entity_index + index );
    */

    // new method: exact match required

    if (ligand_spec == col_arrays.receptors[receptor_index]) {
        // register collision
        int old_val = atomicAdd(col_arrays.counter, 1);
        col_arrays.receptor_ids[old_val] = receptor_index;
        col_arrays.energies[old_val] = col_arrays.ligands[index].energy;

        // delete ligand by setting its message to 0xFFFFFFFF
        col_arrays.ligands[index].emitted_id = 0XFFFFFFFF;
        return true;
    } else {
        // reflect ligand and continue
        reflect_ligand(index, col_arrays, entity_index, dx, dy, velx, vely);
        return true;
    }

}



__global__ void ligand_collision_kernel(uint32_t size, uint32_t search_radius, uint32_t n_receptors, Dim dim, CollisionUtils col_arrays) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // check bounds
    if (i >= size) {
        return;
    }

    if (col_arrays.ligands[i].emitted_id== 0xFFFFFFFF) {
        
        return;
    }


    uint32_t dim_x = dim.x;
    uint32_t dim_y = dim.y;
    uint32_t depth = dim.depth;



    // check for collisions with borders and reflect velocity if necessary
    if (border_collision(col_arrays.ligands, i, dim_x, dim_y)) {
        return; // skip collision detection if border collision occurred
    }

    float x = col_arrays.ligands[i].posx;
    float y = col_arrays.ligands[i].posy;


    // iterate over search area
    for (int dx = -(int)(search_radius); dx <= (int)(search_radius); dx++) {
        // skip if out of bounds
        if ((int)x + dx < 0 || (int)x + dx >= (int)dim_x) {
            continue;
        }
        for (int dy = -(int)(search_radius); dy <= (int)(search_radius); dy++){
            // skip if out of bounds
            if ((int)y + dy < 0 || (int)y + dy >= (int)dim_y) {
                continue;
            }

            // compute grid index: (y * dim_x + x) * depth
            int index = (((int)y + dy) * dim_x + (int)x + dx) * depth;
        
            // check collisions in this cell
            // iterate over depth
            for (int slot = 0; slot < depth; slot++) {
                
                uint32_t entity_index = col_arrays.grid[index + slot];

                if (entity_index == 0xFFFFFFFF) {
                    // empty slot, no more entities in this cell
                    break;
                }


                // check if ligand collides with entity and resolve collision
                if (entity_collision(i, col_arrays, entity_index, n_receptors, x, y)) {
                    return;
                } else {
                    continue; // no collision, check next entity
                }          
            }

        }
    }
}

__global__ void delete_ligands_kernel(LigandCuda* ligands, uint32_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        ligands[i].emitted_id = 0xFFFFFFFF;
    }
}

LigandWrapper error_return(float* energies, uint32_t* receptor_ids, uint32_t* counter) {
    // free device memory
    cudaFree(energies);
    cudaFree(receptor_ids);
    cudaFree(counter);

    LigandWrapper empty_wrapper;
    empty_wrapper.energies = nullptr;
    empty_wrapper.receptor_ids = nullptr;
    empty_wrapper.count = 0;
    return empty_wrapper;
}



extern "C" {
    
    // fills a 3D grid with a specified value
    // pointers are already device pointers
    int fill_grid(uint32_t size, Dim dim, uint32_t* grid, EntityCuda* entities) {

        bool error = false;

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
            printf("Grid launch error: %s\n", cudaGetErrorString(err));
            error = true;
        }

        // copy overflow counter back to host and free device memory
        uint32_t h_overflow;
        cudaMemcpy(&h_overflow, d_overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_overflow);

        if (error) {
            return -1;
        }

        return h_overflow;
    }

    // performs collision detection for ligands against entities in a grid
    // pointers are already device pointers
    LigandWrapper ligand_collision(uint32_t search_radius, Dim dim, uint32_t* grid, EntityCuda* entities, LigandCuda* ligands, uint32_t ligands_size, uint32_t* receptors, uint32_t n_receptors) {

        float* energies;
        uint32_t* receptor_ids;
        uint32_t* counter;

        uint32_t collided_capacity = (uint32_t)ceilf((float)ligands_size * COLLISION_SPACE_FACTOR);

        // allocate memory for collided ligands
        cudaMalloc((void**)&energies, sizeof(float) * collided_capacity);
        cudaMalloc((void**)&receptor_ids, sizeof(uint32_t) * collided_capacity);
        cudaMalloc((void**)&counter, sizeof(uint32_t));
        cudaMemset(counter, 0, sizeof(uint32_t)); // initialize counter to 0

        // define struct to hold all necessary arrays
        CollisionUtils col_arrays;
        col_arrays.grid = grid;
        col_arrays.entities = entities;
        col_arrays.ligands = ligands;
        col_arrays.receptors = receptors;
        col_arrays.energies = energies;
        col_arrays.receptor_ids = receptor_ids;
        col_arrays.counter = counter;
        

        // launch kernel
        uint32_t blockN = (ligands_size + ThreadsPerBlock - 1) / ThreadsPerBlock;
        ligand_collision_kernel<<<blockN, ThreadsPerBlock>>>(ligands_size, search_radius, n_receptors, dim, col_arrays);
        cudaError_t err = cudaDeviceSynchronize(); // wait for kernel to finish

        // check for launch errors
        if (err != cudaSuccess) {
            printf("ligand_collision failed\n");
            printf("Launch error: %s\n", cudaGetErrorString(err));
            return error_return(energies, receptor_ids, counter);
        }

        // copy counter back to host
        uint32_t* h_counter = new uint32_t;
        cudaMemcpy(h_counter, col_arrays.counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (*h_counter > collided_capacity) {
            // overflow, too many collisions
            printf("Error: counter exceeds allocated size: Overflow\n");

            return error_return(energies, receptor_ids, counter);
        }

        // copy collided ligand data back to host
        float* h_energies = (float*)malloc(sizeof(float) * *h_counter);
        uint32_t* h_receptor_ids = (uint32_t*)malloc(sizeof(uint32_t) * *h_counter);
        cudaMemcpy(h_energies, energies, sizeof(float) * *h_counter, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_receptor_ids, receptor_ids, sizeof(uint32_t) * *h_counter, cudaMemcpyDeviceToHost);


        // free device memory
        cudaFree(counter);
        cudaFree(energies);
        cudaFree(receptor_ids);

        // prepare return struct
        LigandWrapper h_collided;
        h_collided.energies = h_energies;
        h_collided.receptor_ids = h_receptor_ids;
        h_collided.count = *h_counter;

        return h_collided;

    }

    // updates ligand positions based on their velocities
    void update_positions(LigandCuda* ligands, uint32_t size, float delta_time) {
        uint32_t blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

        // launch kernel
        update_positions_kernel<<<blockN, ThreadsPerBlock>>>(ligands, size, delta_time);
        cudaError_t err = cudaDeviceSynchronize(); // wait for kernel to finish

        // check for launch errors
        if (err != cudaSuccess) {
            printf("Launch error: %s\n", cudaGetErrorString(err));
        }
    }

    void delete_ligands(LigandCuda* ligands, uint32_t size) {
        uint32_t blockN = (size + ThreadsPerBlock - 1) / ThreadsPerBlock;

        // launch kernel
        delete_ligands_kernel<<<blockN, ThreadsPerBlock>>>(ligands, size);
        cudaError_t err = cudaDeviceSynchronize(); // wait for kernel to finish

        // check for launch errors
        if (err != cudaSuccess) {
            printf("Launch error: %s\n", cudaGetErrorString(err));
        }
    }

}


