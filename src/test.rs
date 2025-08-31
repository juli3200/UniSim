#![cfg(test)]

mod general{
    use crate::world;

    #[test]
    fn create_world(){
        let _world = world::World::new(100);
        
    }

    #[test]
    fn test_movement() {
        let mut world = world::World::new(1);
        world.settings.fps = 1.0;
        world.space.settings.fps = 1.0;

        println!("vel: {:?}", world.entities[0].velocity);

        println!("pos: {:?}", world.entities[0].position);
        world.update();
        println!("pos: {:?}", world.entities[0].position);
        world.update();
        println!("pos: {:?}", world.entities[0].position);
    }
}

mod io_tests {
    use crate::world;
    use crate::*;

    #[test]
    fn test_save(){
        let n = 10000;
        let mut world = world::World::new(100);
        edit_settings!(&mut world, fps = 60.0, velocity = 3.0);


        let e = world.save("alpha.bin");
        println!("Save result: {:?}", e);
        for _ in 0..n {
            world.update();
        }

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