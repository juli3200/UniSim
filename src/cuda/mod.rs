#![cfg(feature = "cuda")]
use crate::{objects, settings_::Settings};
use crate::objects::{Entity, Ligand};

pub(crate) mod cuda_bindings;
pub(crate) mod cudaworld;

const EXTRA_SPACE_ENTITY: f32 = 1.2; // allocate 20% more space than needed for entities
const EXTRA_SPACE_LIGAND: f32 = 2.0; // allocate 100% more space than needed for ligands
const MIN_SPACE_LIGAND: usize = 1_000_000; // minimum space for ligands
const BUFFER_SIZE: usize = 10 * 1024 * 1024; // 10 MB buffer for saving data from GPU (remove const later)

/// CUDA-related structures and functions
/// e.g as CUDA arrays, kernels links, etc.

#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct CollisionArraysHost {
    pub(crate) collided_message: *mut u32,
    pub(crate) collided_pos: *mut f32,
    pub(crate) collided_entities: *mut u32,
    pub(crate) counter: u32,
}


#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct EntityArrays {
    pos: *mut f32,
    size: *mut f32,
    id: *mut u32,
    num_entities: usize,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct LigandArrays {
    pub(crate) pos: *mut f32,
    pub(crate) vel: *mut f32,
    pub(crate) message: *mut u32,

    pub(crate) num_ligands: usize,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct Dim{
    x: u32,
    y: u32,
    depth: u32,
}

// if gpu is active positions, velocities, and sizes of the objects are each stored in a single array on gpu memory
// this is to minimize the number of memory transfers between cpu and gpu
#[derive(Debug, Clone)]
pub(crate) struct CUDAWorld{

    pub(crate) settings: Settings,

    // The following are pointers to CUDA device memory

    // 3D Array: grid[x][y][i] = index of the ligand at that position, or 0 if empty
    // the third dimension is a list of indices, to allow multiple ligands in the same cell
    // the size of the third dimension is fixed, settings.cuda_slots_per_cell
    grid: *mut u32,

    // Buffer to save data from GPU to CPU
    save_buffer: *mut u8,

    // capacity of entities
    entity_cap: u32,

    // Entity data arrays
    entities: EntityArrays,

    // capacity of ligands
    ligand_cap: u32,

    // Ligand data arrays
    ligands: LigandArrays,
}

// Test functions to access private fields
#[allow(dead_code)]
#[cfg(test)]
impl CUDAWorld {
    pub(crate) fn get_entity_arrays(&self) -> EntityArrays {
        self.entities.clone()
    }

    pub(crate) fn get_ligand_arrays(&self) -> LigandArrays {
        self.ligands.clone()
    }
}