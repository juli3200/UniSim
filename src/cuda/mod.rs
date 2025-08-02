use ndarray::Array1;

/// CUDA-related structures and functions
/// e.g as CUDA arrays, kernels links, etc.

// if gpu is active positions, velocities, and sizes of the objects are stored in a single array
#[derive(Debug, Clone)]
pub struct CUDAWorld {
    pub(crate) entities_pos: Array1<f32>,
    pub(crate) entities_vel: Array1<f32>,
    pub(crate) entities_size: Array1<f32>,
    pub(crate) entities_id: Array1<u32>,

    pub(crate) ligands_pos: Array1<f32>,
    pub(crate) ligands_vel: Array1<f32>,
}
