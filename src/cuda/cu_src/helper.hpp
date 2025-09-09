#pragma once

// helper.hpp
// Utility definitions for CUDA source files


// Add your helper function declarations below

#define u_int unsigned int

typedef struct{
    float* pos; // 2 floats per ligand
    float* vel; // 2 floats per ligand
    u_int* message; // 1 u_int per ligand
    size_t num_ligands; // number of ligands
} LigandArrays;

typedef struct{
    float* pos; // 2 floats per entity
    float* vel; // 2 floats per entity
    float* acc; // 2 floats per entity
    float* size; // 2 floats per entity
    u_int* id;  // 1 int per entity

    size_t num_entities; // number of entities

} EntityArrays;

typedef struct{
    u_int* collided_message; // stores messages of collided ligands
    float* collided_pos; // stores velocities of ligands
    u_int* collided_entities; // stores ids of collided entities
    u_int* counter; // number of collisions
} CollisionArraysDevice;

// ------------------------------------ Hosting array structures to give back to Rust ------------------------------------
    
typedef struct{
    u_int* collided_message;
    float* collided_pos;
    u_int* collided_entities;
    u_int counter;
}CollisionArraysHost;

