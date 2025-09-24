#pragma once

// helper.hpp
// Utility definitions for CUDA source files


// Add your helper function declarations below

#define u_long unsigned int

typedef struct{
    float* pos; // 2 floats per ligand
    float* vel; // 2 floats per ligand
    u_long* message; // 1 u_long per ligand
    size_t num_ligands; // number of ligands
} LigandArrays;

typedef struct{
    float* pos; // 2 floats per entity
    float* size; // 2 floats per entity
    u_long* id;  // 1 int per entity
    size_t num_entities; // number of entities

} EntityArrays;

typedef struct{
    u_long* collided_message; // stores messages of collided ligands
    float* collided_pos; // stores velocities of ligands
    u_long* collided_entities; // stores ids of collided entities
    u_long* counter; // number of collisions
} CollisionArraysDevice;

// ------------------------------------ Hosting array structures to give back to Rust ------------------------------------
    
typedef struct{
    u_long* collided_message;
    float* collided_pos;
    u_long* collided_entities;
    u_long counter;
}CollisionArraysHost;

typedef struct{
    u_long x;
    u_long y;
    u_long depth;
} Dim;