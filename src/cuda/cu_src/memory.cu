#include <iostream>
#include <cuda_runtime.h>

#include "helper.hpp"

__global__ void remove_dead_ligands_kernel(LigandCuda* new_ligands, LigandCuda* old_ligands, uint32_t size, uint32_t* counter) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (old_ligands[idx].emitted_id != 0xFFFFFFFF) {
        uint32_t new_idx = atomicAdd(counter, 1);
        new_ligands[new_idx] = old_ligands[idx];
    }
}


extern "C"{
    // ----------------float memory management functions----------------

    // allocates memory on the GPU and returns a pointer to it
    float* alloc_f(uint32_t size){
        float* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(float));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_f(float* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_f(float* d_ptr, float* h_ptr, uint32_t size){
        cudaMemcpy(d_ptr, h_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_f(float* h_ptr, float* d_ptr, uint32_t size){
        cudaMemcpy(h_ptr, d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_f(float* target, float* origin, uint32_t size){
        cudaMemcpy(target, origin, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }


    // ----------------uint32_t memory management functions----------------

    // allocates memory on the GPU and returns a pointer to it
    uint32_t* alloc_u32(uint32_t size){
        uint32_t* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(uint32_t));

        return d_ptr;
    }

    // frees memory on the GPU
    void free_u32(uint32_t* d_ptr){
        cudaFree(d_ptr);
    }

    // copies memory from host to device
    void copy_HtoD_u32(uint32_t* d_ptr, uint32_t* h_ptr, uint32_t size){
        cudaMemcpy(d_ptr, h_ptr, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }

    // copies memory from device to host
    void copy_DtoH_u32(uint32_t* h_ptr, uint32_t* d_ptr, uint32_t size){
        cudaMemcpy(h_ptr, d_ptr, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    // copies memory from device to device
    void copy_DtoD_u32(uint32_t* target, uint32_t* origin, uint32_t size){
        cudaMemcpy(target, origin, size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    }



    // ----------------EntityArray memory management functions----------------
    EntityCuda* alloc_entity(uint32_t size){
        EntityCuda* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(EntityCuda));

        return d_ptr;
    }

   void free_entity(EntityCuda* d_ptr){
        cudaFree(d_ptr);
    }

    void copy_HtoD_entity(EntityCuda* d_ptr, EntityCuda* h_ptr, uint32_t size){
        cudaMemcpy(d_ptr, h_ptr, size * sizeof(EntityCuda), cudaMemcpyHostToDevice);
    }

    void copy_DtoH_entity(EntityCuda* h_ptr, EntityCuda* d_ptr, uint32_t size){
        cudaMemcpy(h_ptr, d_ptr, size * sizeof(EntityCuda), cudaMemcpyDeviceToHost);
    }

    void copy_DtoD_entity(EntityCuda* target, EntityCuda* origin, uint32_t size){
        cudaMemcpy(target, origin, size * sizeof(EntityCuda), cudaMemcpyDeviceToDevice);
    }

    // ----------------LigandArray memory management functions----------------
    LigandCuda* alloc_ligand(uint32_t size){
        LigandCuda* d_ptr;
        cudaMalloc((void**)&d_ptr, size * sizeof(LigandCuda));

        return d_ptr;
    }

    void free_ligand(LigandCuda* d_ptr){
          cudaFree(d_ptr);
     }

    void copy_HtoD_ligand(LigandCuda* d_ptr, LigandCuda* h_ptr, uint32_t size){
        cudaMemcpy(d_ptr, h_ptr, size * sizeof(LigandCuda), cudaMemcpyHostToDevice);
    }

    void copy_DtoH_ligand(LigandCuda* h_ptr, LigandCuda* d_ptr, uint32_t size){
        cudaMemcpy(h_ptr, d_ptr, size * sizeof(LigandCuda), cudaMemcpyDeviceToHost);
    }

    void copy_DtoD_ligand(LigandCuda* target, LigandCuda* origin, uint32_t size){
        cudaMemcpy(target, origin, size * sizeof(LigandCuda), cudaMemcpyDeviceToDevice);
    }


    
    void clear_grid(uint32_t* grid, uint32_t size) {
        // set all grid values to 0xFFFFFFFF
        cudaMemset(grid, 0xFFFFFFFF, size * sizeof(uint32_t));
    }

    Wrapper remove_dead_values(LigandCuda* old_ligands, uint32_t size) {
        LigandCuda* new_ligands;
        cudaMalloc((void**)&new_ligands, size * sizeof(LigandCuda));

        uint32_t* d_counter;
        cudaMalloc((void**)&d_counter, sizeof(uint32_t));
        cudaMemset(d_counter, 0, sizeof(uint32_t));

        uint32_t threads_per_block = 256;
        uint32_t blocks = (size + threads_per_block - 1) / threads_per_block;

        remove_dead_ligands_kernel<<<blocks, threads_per_block>>>(new_ligands, old_ligands, size, d_counter);
        cudaDeviceSynchronize();

        uint32_t h_counter;
        cudaMemcpy(&h_counter, d_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_counter);
        cudaFree(old_ligands);

        Wrapper wrapper;
        wrapper.ligands = new_ligands; 
        wrapper.ligand_count = h_counter;

        return wrapper;
    }

}