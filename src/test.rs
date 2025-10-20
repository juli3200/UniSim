#![cfg(all(test, feature = "debug"))]

mod general{

    use crate::prelude::*;


    #[test]
    fn create_world(){
        let _world = World::new(settings!(100, spawn_size = 5.0));
        
    }

    #[test]
    fn test_movement() {
        let mut world = World::default();


        println!("vel: {:?}", world.entities[0].velocity);

        println!("pos: {:?}", world.entities[0].position);
        world.update();
        println!("pos: {:?}", world.entities[0].position);
        world.update();
        println!("pos: {:?}", world.entities[0].position);
    }

    #[test]
    fn collision_test() {
        let mut world = World::new(settings!(2, spawn_size = 0.5, give_start_vel = false, velocity = 1.0, dimensions = (10,10), fps = 30.0));
        world.entities[0].size = 2.0;
        world.space.max_size = 2.0;
        world.entities[0].position = ndarray::Array1::from(vec![3.0, 3.0]);
        world.entities[0].velocity = ndarray::Array1::from(vec![0.0, 0.0]);
        world.entities[1].position = ndarray::Array1::from(vec![8.0, 3.0]);
        world.entities[1].velocity = ndarray::Array1::from(vec![-1.0, 0.0]);

        let e = world.save("testfiles/col.bin");
        if let Err(e) = e {
            eprintln!("Error saving world: {}", e);
        }
        world.run(1024);


    }

    #[test]
    fn ligand_test(){
        let setting = settings!(1, spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (10,10), give_start_vel = true);
        let mut world = World::new(setting);

        // add ligands manually
        world.add_ligands(1000);

        world.save("ligand_test.bin").expect("Failed to save world");
        world.run(1024);


    }

    #[test]
    fn gravity_test(){
        let setting = settings!(100, velocity = 3.0, dimensions = (100,100), gravity = vec![0.0, -1.0], store_capacity = 1000);
        let mut world = World::new(setting);

        world.save("testfiles/gravity_test.bin").expect("Failed to save world");

        world.run(1000);

    }

}

mod io_tests {
    use crate::prelude::*;


    #[test]
    fn test_save(){
        let n = 10000;
        let mut world = World::new(settings!(100, spawn_size = 5.0));
        edit_settings!(&mut world, fps = 60.0, velocity = 3.0);


        let e = world.save("testfiles/alpha.bin");
        println!("Save result: {:?}", e);
        world.run(n);

    }
}

mod dna_tests {
    use crate::prelude::*;


    #[test]
    fn test_tumble(){
        let mut world = World::new(settings!(100, fps = 40.0, velocity = 3.0, store_capacity = 2000, tumble_chance = 0.3));
        
        edit_settings!(&mut world, drag = 0.1, gravity = vec![0.0, -0.2]);

        world.save("testfiles/tumble_test.bin").expect("Failed to save world");
        world.run(2000);


    }
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use crate::prelude::*;
    
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

    #[cfg(feature = "cuda")]
    #[test]
    fn ligands_test(){


        let mut world = World::new(settings!(1, spawn_size = 1.0, fps = 10.0, velocity = 3.0, dimensions = (10,10), give_start_vel = true, store_capacity = 100));
        world.cuda_initialize().expect("Init expect");
        world.save("testfiles/ligands_test.bin").expect("Save expect");


        // add ligands manually
        /*
        for i in 0..9 {
            world.ligands.push(objects::Ligand {
                id: i,
                position: Array1::from_vec(vec![i as f32 + 0.1, 1.0]),
                velocity: Array1::from_vec(vec![0.0, 1.0]),
                message: i as u32 + 1,
            });
            world.ligands_count += 1;
        }*/
        world.add_ligands(1000);

        world.run(100);
    }


}





