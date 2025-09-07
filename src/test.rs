#![cfg(test)]

mod general{
    use crate::{settings, world};

    #[test]
    fn create_world(){
        let _world = world::World::new(settings!(100, spawn_size = 5.0));
        
    }

    #[test]
    fn test_movement() {
        let mut world = world::World::default();


        println!("vel: {:?}", world.entities[0].velocity);

        println!("pos: {:?}", world.entities[0].position);
        world.update();
        println!("pos: {:?}", world.entities[0].position);
        world.update();
        println!("pos: {:?}", world.entities[0].position);
    }

    #[test]
    fn collision_test() {
        let mut world = world::World::new(settings!(2, spawn_size = 0.5, give_start_vel = false, velocity = 1.0, dimensions = (10,10), fps = 30.0));
        world.entities[0].size = 2.0;
        world.space.max_size = 2.0;
        world.entities[0].position = ndarray::Array1::from(vec![3.0, 3.0]);
        world.entities[0].velocity = ndarray::Array1::from(vec![0.0, 0.0]);
        world.entities[1].position = ndarray::Array1::from(vec![8.0, 3.0]);
        world.entities[1].velocity = ndarray::Array1::from(vec![-1.0, 0.0]);

        let e = world.save("col.bin");
        if let Err(e) = e {
            eprintln!("Error saving world: {}", e);
        }
        world.run(1024);


    }
}

mod io_tests {
    use crate::world;
    use crate::*;

    #[test]
    fn test_save(){
        let n = 10000;
        let mut world = world::World::new(settings!(100, spawn_size = 5.0));
        edit_settings!(&mut world, fps = 60.0, velocity = 3.0);


        let e = world.save("alpha.bin");
        println!("Save result: {:?}", e);
        world.run(n);

    }
}


mod cuda_tests {

    #[cfg(feature = "cuda")]
    use crate::cuda::cuda_bindings::tests_gpu as cb;

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