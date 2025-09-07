
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
        pub(crate) fn alloc_f(i: *mut f32, size: i32) -> *mut f32;
        pub(crate) fn release_f(d_ptr: *mut f32) -> i32;
        pub(crate) fn copy_HtoD_f(d_ptr: *mut f32, h_ptr: *mut f32, size: i32);
        pub(crate) fn copy_DtoH_f(h_ptr: *mut f32, d_ptr: *mut f32, size: i32);

        // uint32 memory management functions
        pub(crate) fn alloc_u(i: *mut u32, size: i32) -> *mut u32;
        pub(crate) fn release_u(d_ptr: *mut u32) -> i32;
        pub(crate) fn copy_HtoD_u(d_ptr: *mut u32, h_ptr: *mut u32, size: i32);
        pub(crate) fn copy_DtoH_u(h_ptr: *mut u32, d_ptr: *mut u32, size: i32);
    }
}

pub(crate) mod grid_gpu{
    unsafe extern "C" {
        pub(crate) fn clear_grid(grid: *mut u32, size: i32) -> i32;
        pub(crate) fn fill_grid(
            size: i32,
            dim: *mut i32,
            grid: *mut u32,
            posx: *mut f32,
            posy: *mut f32,
            cell: *mut f32,
        ) -> i32;

    }
}