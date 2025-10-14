#![allow(dead_code)]


pub(crate) mod tests_gpu{
    unsafe extern "C" {
        pub(crate) fn cuda_test() -> i32;
        pub(crate) fn alloc_memory_cu(i: i32) -> *mut i32;
        pub(crate) fn release_memory_cu(d_ptr: *mut i32) -> i32;
    }
}

pub(crate) mod memory_gpu{
    use crate::cuda::*;
    unsafe extern "C" {
        // float memory management functions
        pub(crate) fn alloc_f(size: u32) -> *mut f32;
        pub(crate) fn free_f(d_ptr: *mut f32) -> i32;
        pub(crate) fn copy_HtoD_f(d_ptr: *mut f32, h_ptr: *const f32, size: u32);
        pub(crate) fn copy_DtoH_f(h_ptr: *mut f32, d_ptr: *const f32, size: u32);
        pub(crate) fn copy_DtoD_f(target: *mut f32, origin: *const f32, size: u32);
        pub(crate) fn clear_f(d_ptr: *mut f32, size: u32);

        // uint32 memory management functions
        pub(crate) fn alloc_u16(size: u32) -> *mut u16;
        pub(crate) fn free_u16(d_ptr: *mut u16) -> i32;
        pub(crate) fn copy_HtoD_u16(d_ptr: *mut u16, h_ptr: *const u16, size: u32);
        pub(crate) fn copy_DtoH_u16(h_ptr: *mut u16, d_ptr: *const u16, size: u32);
        pub(crate) fn copy_DtoD_u16(target: *mut u16, origin: *const u16, size: u32);
        pub(crate) fn clear_u16(d_ptr: *mut u16, size: u32);

        // EntityCuda memory management functions
        pub(crate) fn alloc_entity(size: u32) -> *mut EntityCuda;
        pub(crate) fn free_entity(d_ptr: *mut EntityCuda) -> i32;
        pub(crate) fn copy_HtoD_entity(d_ptr: *mut EntityCuda, h_ptr: *const EntityCuda, size: u32);
        pub(crate) fn copy_DtoH_entity(h_ptr: *mut EntityCuda, d_ptr: *const EntityCuda, size: u32);
        pub(crate) fn copy_DtoD_entity(target: *mut EntityCuda, origin: *const EntityCuda, size: u32);
        pub(crate) fn clear_entity(d_ptr: *mut EntityCuda, size: u32);

        // LigandCuda memory management functions
        pub(crate) fn alloc_ligand(size: u32) -> *mut LigandCuda;
        pub(crate) fn free_ligand(d_ptr: *mut LigandCuda) -> i32;
        pub(crate) fn copy_HtoD_ligand(d_ptr: *mut LigandCuda, h_ptr: *const LigandCuda, size: u32);
        pub(crate) fn copy_DtoH_ligand(h_ptr: *mut LigandCuda, d_ptr: *const LigandCuda, size: u32);
        pub(crate) fn copy_DtoD_ligand(target: *mut LigandCuda, origin: *const LigandCuda, size: u32);
        pub(crate) fn clear_ligand(d_ptr: *mut LigandCuda, size: u32);


    }
}

pub(crate) mod grid_gpu{
    use crate::cuda::*;
    unsafe extern "C" {
        pub(crate) fn fill_grid(size: u32, dim: Dim, grid: *mut u16, entities: *const EntityCuda) -> i32;
        pub(crate) fn ligand_collision(search_radius: u32, dim: Dim, grid: *mut u16, entities: *const EntityCuda, 
            ligands: *const LigandCuda, ligands_size: u32, receptors: *const u16, n_receptors: u32) -> LigandWrapper; 
        pub(crate) fn update_positions(l_arrays: *mut LigandCuda, size: u32, delta_time: f32);
    }
}