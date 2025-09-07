#![cfg(feature = "cuda")]
use crate::Settings;


pub(crate) mod cuda_bindings;


/// CUDA-related structures and functions
/// e.g as CUDA arrays, kernels links, etc.

// if gpu is active positions, velocities, and sizes of the objects are each stored in a single array on gpu memory
// this is to minimize the number of memory transfers between cpu and gpu
#[derive(Debug, Clone)]
pub(crate) struct CUDAWorld{

    pub(crate) settings: Settings,

    // The following are pointers to CUDA device memory


    // 3D Array: grid[x][y] [i] = index of the ligand at that position, or 0 if empty
    // the third dimension is a list of indices, to allow multiple ligands in the same cell
    // the size of the third dimension is fixed, settings.cuda_slots_per_cell
    pub(crate) grid: *mut u32,

    // Buffer to save data from GPU to CPU
    pub(crate) save_buffer: *mut u8,

    // Entity data arrays
    pub(crate) entities_pos: *mut f32,
    pub(crate) entities_vel: *mut f32,
    pub(crate) entities_size: *mut f32,
    pub(crate) entities_id: *mut u32,

    // Ligand data arrays
    pub(crate) ligands_pos: *mut f32,
    pub(crate) ligands_vel: *mut f32,
}

impl CUDAWorld {
    pub(crate) fn new(settings: &Settings) -> Self {
        Self {
            settings: settings.clone(),

            grid: std::ptr::null_mut(),

            save_buffer: std::ptr::null_mut(),

            entities_pos: std::ptr::null_mut(),
            entities_vel: std::ptr::null_mut(),
            entities_size: std::ptr::null_mut(),
            entities_id: std::ptr::null_mut(),

            ligands_pos: std::ptr::null_mut(),
            ligands_vel: std::ptr::null_mut(),
        }
    }
}