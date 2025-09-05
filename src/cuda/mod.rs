#![cfg(feature = "cuda")]
use ndarray::{Array1, Array3};
use crate::settings_::Settings;


pub(crate) mod cuda_bindings;


/// CUDA-related structures and functions
/// e.g as CUDA arrays, kernels links, etc.

// if gpu is active positions, velocities, and sizes of the objects are stored in a single array
#[derive(Debug, Clone)]
pub struct CUDAWorld {

    pub(crate) settings: Settings,

    // 3D Array: grid[x][y] [i] = index of the ligand at that position, or 0 if empty
    // the third dimension is a list of indices, to allow multiple ligands in the same cell
    // the size of the third dimension is fixed, settings.cuda_slots_per_cell
    pub(crate) grid: Array3<u32>,


    pub(crate) entities_pos: Array1<f32>,
    pub(crate) entities_vel: Array1<f32>,
    pub(crate) entities_size: Array1<f32>,
    pub(crate) entities_id: Array1<u32>,

    pub(crate) ligands_pos: Array1<f32>,
    pub(crate) ligands_vel: Array1<f32>,
}
