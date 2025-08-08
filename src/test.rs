#![cfg(test)]


mod tests {

    use crate::cuda::cuda_bindings as cb;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_function() {
        
        unsafe {
            let result = cb::cuda_test();
            println!("CUDA test function returned: {}", result);
        }
    }

    #[test]
    fn test_memory_allocation() {
        unsafe {
            let d_ptr = cb::alloc_memory_cu(1000);

            let value = cb::release_memory_cu(d_ptr);
            println!("value: {}", value);
        }
    }
}