#pragma once

#include <cstdint>
// helper.hpp
// Utility definitions for CUDA source files


typedef struct{
    uint32_t emitted_id; // id of the entity that emitted the ligand
    float posx; // x position of the ligand
    float posy; // y position of the ligand
    float velx; // x velocity of the ligand
    float vely; // y velocity of the ligand

    uint32_t spec; // ligand spec number
    float energy; // ligand energy
} LigandCuda;

typedef struct{
    float posx; // x position of the entity
    float posy; // y position of the entity
    float velx; // x velocity of the entity
    float vely; // y velocity of the entity
    float size; // size of the entity
    uint32_t id; // entity id

} EntityCuda;

typedef struct{
    uint32_t x;
    uint32_t y;
    uint32_t depth;
} Dim;

typedef struct{
    uint32_t* receptor_ids; // array of receptor ids that collided with ligands
    float* energies; // array of energies of the collided ligands
    uint32_t count; // number of collided ligands
} LigandWrapper;

// struct to hold all necessary arrays for collision detection
typedef struct {
    uint32_t* grid;
    EntityCuda* entities;
    LigandCuda* ligands;
    uint32_t* receptors;

    float* energies;
    uint32_t* receptor_ids; // ids of receptors that were bound in receptor array
    // entity ids of collided entities can be computed as receptor_ids[i] / n_receptors
    uint32_t* counter;

} CollisionUtils;