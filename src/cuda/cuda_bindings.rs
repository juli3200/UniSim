
pub(crate) mod tests_gpu{
    unsafe extern "C" {
        pub(crate) fn cuda_test() -> i32;
        pub(crate) fn alloc_memory_cu(i: i32) -> *mut i32;
        pub(crate) fn release_memory_cu(d_ptr: *mut i32) -> i32;
    }
}

pub(crate) mod memory_gpu{
    unsafe extern "C" {
        // float memory management functions
        pub(crate) fn alloc_f(size: u32) -> *mut f32;
        pub(crate) fn free_f(d_ptr: *mut f32) -> i32;
        pub(crate) fn copy_HtoD_f(d_ptr: *mut f32, h_ptr: *mut f32, size: u32);
        pub(crate) fn copy_DtoH_f(h_ptr: *mut f32, d_ptr: *mut f32, size: u32);
        pub(crate) fn copy_DtoD_f(target: *mut f32, origin: *mut f32, size: u32);
        pub(crate) fn clear_f(d_ptr: *mut f32, size: u32);

        // uint32 memory management functions
        pub(crate) fn alloc_u(size: u32) -> *mut u32;
        pub(crate) fn free_u(d_ptr: *mut u32) -> i32;
        pub(crate) fn copy_HtoD_u(d_ptr: *mut u32, h_ptr: *mut u32, size: u32);
        pub(crate) fn copy_DtoH_u(h_ptr: *mut u32, d_ptr: *mut u32, size: u32);
        pub(crate) fn copy_DtoD_u(target: *mut u32, origin: *mut u32, size: u32);
        pub(crate) fn clear_u(d_ptr: *mut u32, size: u32);

        // char memory management functions
        pub(crate) fn alloc_c(size: u32) -> *mut u8;
        pub(crate) fn free_c(d_ptr: *mut u8) -> i32;
        pub(crate) fn copy_HtoD_c(d_ptr: *mut u8, h_ptr: *mut u8, size: u32);
        pub(crate) fn copy_DtoH_c(h_ptr: *mut u8, d_ptr: *mut u8, size: u32);   
        pub(crate) fn copy_DtoD_c(target: *mut u8, origin: *mut u8, size: u32);
        pub(crate) fn clear_c(d_ptr: *mut u8, size: u32);
    }
}

pub(crate) mod grid_gpu{
    use crate::cuda::*;
    unsafe extern "C" {
        pub(crate) fn fill_grid(size: u32, dim: *mut u32, grid: *mut u32, pos: *mut f32, cell: *mut u32,) -> i32;
        pub(crate) fn ligand_collision(search_radius: f32, dim: *mut u32, grid: *mut u32, e_arrays: EntityArrays, l_arrays: LigandArrays) -> CollisionArraysHost;

    }
}