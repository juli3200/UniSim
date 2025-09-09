#pragma once

// helper.hpp
// Utility definitions for CUDA source files


// Add your helper function declarations below

#define u_int unsigned int

struct LigandArrays{
    float* pos_ligand; // 2 floats per ligand
    float* vel_ligand; // 2 floats per ligand
    u_int* message_ligand; // 1 u_int per ligand
};

struct EntityArrays{
    float* pos_entity; // 2 floats per entity
    float* vel_entity; // 2 floats per entity
    float* acc_entity; // 2 floats per entity
    float* size_entity; // 2 floats per entity
    u_int* id_entity;  // 1 int per entity

};

struct CollisionArrays{
    u_int* collided_message; // stores messages of collided ligands
    float* collided_pos; // stores velocities of ligands
    u_int* collided_entities; // stores ids of collided entities
    u_int* counter; // counts number of collisions
};

// Hosting array structures to give back to Rust
struct FloatArray{
    float* data;
    u_int size;
};

struct UIntArray{
    u_int* data;
    u_int size;
};

struct CharArray{
    char* data;
    u_int size;
};