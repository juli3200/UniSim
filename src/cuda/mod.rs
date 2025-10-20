#![cfg(feature = "cuda")]
use crate::settings_::Settings;
use crate::objects::{Entity, Ligand};

pub(crate) mod cuda_bindings;
pub(crate) mod cudaworld;

const EXTRA_SPACE_ENTITY: f32 = 1.2; // allocate 20% more space than needed for entities
const EXTRA_SPACE_LIGAND: f32 = 2.0; // allocate 100% more space than needed for ligands
const MIN_SPACE_LIGAND: usize = 1_000_000; // minimum space for ligands

/// CUDA-related structures and functions
/// e.g as CUDA arrays, kernels links, etc.

#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct EntityCuda {
    posx: f32, // x position of the entity
    posy: f32, // y position of the entity
    velx: f32, // x velocity of the entity
    vely: f32, // y velocity of the entity
    size: f32, // size of the entity
    id: u32,   // id of the entity

    // receptors are stored as pointer because it does not change 
    // EntityCuda is copied to GPU memory every step
    
}

#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct LigandCuda {
    emitted_id: u32, // id of the entity that emitted the ligand

    posx: f32, // x position of the ligand
    posy: f32, // y position of the ligand
    velx: f32, // x velocity of the ligand
    vely: f32, // y velocity of the ligand

    spec: u32,
    energy: f32,

}

#[repr(C)]
pub(crate) struct Dim{
    x: u32,
    y: u32,
    depth: u32,
}

#[repr(C)]
pub(crate) struct LigandWrapper{
    pub(crate) receptor_ids: *mut u32,
    pub(crate) energies: *mut f32,
    pub(crate) count: u32,
}


pub(crate) enum IncreaseType{
    Entity,
    Ligand,
    Grid
}
// if gpu is active positions, velocities, and sizes of the objects are each stored in a single array on gpu memory
// this is to minimize the number of memory transfers between cpu and gpu
#[derive(Debug, Clone)]
pub struct CUDAWorld{

    pub settings: Settings,

    // The following are pointers to CUDA device memory

    // 3D Array: grid[x][y][i] = index of the ligand at that position, or 0 if empty
    // the third dimension is a list of indices, to allow multiple ligands in the same cell
    // the size of the third dimension is fixed, settings.cuda_slots_per_cell
    grid: *mut u32,


    entity_cap: u32,
    entity_count: u32,

    // Entity data arrays
    entities: *mut EntityCuda,

    // receptor data arrays
    receptors: *mut u32, // size: entity_cap * settings.receptor_capacity() (only spec stored)
    

    ligand_cap: u32,
    pub(crate) ligand_count: u32,

    // Ligand data arrays
    pub(crate) ligands: *mut LigandCuda,
}
