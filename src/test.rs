#![cfg(test)]

mod general{
    use crate::world;

    #[test]
    fn create_world(){
        let _world = world::World::new();
        
    }
}

mod io_tests {
    use crate::world;

    #[test]
    fn test_save(){
        let mut world = world::World::new();
        let _ = world.save("alpha.bin");
    }
}


mod cuda_tests {

    use crate::cuda::cuda_bindings::tests as cb;

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_function() {
        
        unsafe {
            let result = cb::cuda_test();
            println!("CUDA test function returned: {}", result);
        }
    }


    // Test if i can store memory and access it later
    #[cfg(feature = "cuda")]
    #[test]
    fn test_memory_allocation() {
        unsafe {
            let d_ptr = cb::alloc_memory_cu(1000);

            let value = cb::release_memory_cu(d_ptr);
            println!("value: {}", value);
        }
    }
}