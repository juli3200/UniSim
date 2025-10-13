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

    uint16_t spec; // ligand spec number
    float energy; // ligand energy
} LigandCuda;

typedef struct{
    float posx; // x position of the entity
    float posy; // y position of the entity
    float velx; // x velocity of the entity
    float vely; // y velocity of the entity
    float size; // size of the entity
    uint32_t spec; // entity spec number

} EntityCuda;

typedef struct{
    uint32_t x;
    uint32_t y;
    uint32_t depth;
} Dim;