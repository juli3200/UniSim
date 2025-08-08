
unsafe extern "C" {
    pub(crate) fn cuda_test() -> i32;
    pub(crate) fn alloc_memory_cu(i: i32) -> *mut i32;
    pub(crate) fn release_memory_cu(d_ptr: *mut i32) -> i32;
}