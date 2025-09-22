#![cfg(test)]

use crate::world::World;
use ndarray::Array1;
use crate::objects;
use crate::settings;
use rand::Rng;


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

    #[test]
    fn ligand_test(){
        let setting = settings!(1, spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (10,10), give_start_vel = true);
        let mut world = world::World::new(setting);

        // add ligands manually
        world.add_ligands(1000);

        world.save("ligand_test.bin").expect("Failed to save world");
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


#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    
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


        let mut world = World::new(settings!(1, spawn_size = 1.0, fps = 10.0, velocity = 3.0, dimensions = (10,10), give_start_vel = true, store_capacity = 1000));
        world.cuda_initialize().expect("Init expect");
        world.save("ligands_test.bin").expect("Save expect");


        // add ligands manually
        for i in 0..9 {
            world.ligands.push(objects::Ligand {
                id: i,
                position: Array1::from_vec(vec![i as f32 + 0.1, 1.0]),
                velocity: Array1::from_vec(vec![0.0, 1.0]),
                message: i as u32 + 1,
            });
            world.ligands_count += 1;
        }

        world.run(1000);
    }
}


mod performance_tests {
    use std::time::{Duration, Instant};
    use super::*;


    fn test_performance_cpu(n : usize, ligands: usize) -> Duration {


        let mut world = World::new(settings!(1000, spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (100,100), give_start_vel = true));

        world.add_ligands(ligands);

        let start = Instant::now();
        world.run(n);
        let duration = start.elapsed();

        return duration;

    }

    #[cfg(feature = "cuda")]
    fn test_performance_gpu(n : usize, ligands: usize) -> Duration {


        let mut world = World::new(settings!(1000, spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (100,100), give_start_vel = true));
        world.cuda_initialize().expect("Failed to initialize CUDA");

        world.add_ligands(ligands);

        let start = Instant::now();
        world.run(n);
        let duration = start.elapsed();

        return duration;

    }

    // not working properly yet
    #[test]
    #[cfg(feature = "cuda")]
    fn test_compare_performance() {
        let n = 1000;
        let mut cpu_durations = Vec::new();
        let mut gpu_durations = Vec::new();
        for ligands in [10, 100, 500, 1000, 5000, 10000].iter() {
            println!("Testing with {} ligands", ligands);
            println!("CPU:");
            let cpu_duration = test_performance_cpu(n, *ligands);
            cpu_durations.push(cpu_duration);
            println!("CPU Duration: {:?}", cpu_duration);
            println!("GPU:");
            let gpu_duration = test_performance_gpu(n, *ligands);
            gpu_durations.push(gpu_duration);
            println!("GPU Duration: {:?}", gpu_duration);
            println!("-----------------------");
        }

        // store results in csv file
        let mut wtr = csv::Writer::from_path("performance_results.csv").expect("Failed to create CSV writer");
        wtr.write_record(&["Ligands", "CPU Duration (ms)", "GPU Duration (ms)"]).expect("Failed to write header");
        for i in 0..cpu_durations.len() {
            wtr.write_record(&[
                (10_usize.pow(i as u32 + 1)).to_string(),
                cpu_durations[i].as_millis().to_string(),
                gpu_durations[i].as_millis().to_string(),
            ]).expect("Failed to write record");
        }
        wtr.flush().expect("Failed to flush CSV writer");
        println!("Performance results saved to performance_results.csv");

    }
}


impl World {
    // add n ligands at random positions
    // only for testing purposes
    pub fn add_ligands(&mut self, n: usize) {

        let mut rng = rand::rng();

        for _ in 0..n {
            let x = rng.random_range(0.0..self.space.width as f32);
            let y = rng.random_range(0.0..self.space.height as f32);
            let position = Array1::from_vec(vec![x, y]);


            let len = (position[0].powi(2) + position[1].powi(2)).sqrt();
            let norm_pos = Array1::from_vec(vec![position[0]/len, position[1]/len]);
            // add ligand at random position
            // ensure position is within bounds
            let ligand = objects::Ligand {
                id: self.counter,
                position,
                velocity: norm_pos, // velocity is not tracked after collision
                message: 0u32,
            };
            self.counter += 1;
            self.ligands.push(ligand);
            self.ligands_count += 1;
        }
    }
}

// Test debugging impl block for World
#[cfg(feature = "cuda")]
impl World{

    pub(crate) fn copy_ligands(&mut self, positions: &[f32], messages: &[u32], len: usize){
        {   
            // print the received ligands
            let mut ligands = vec![];
            for i in 0..len {
                use crate::objects::Ligand;

                let ligand = Ligand {
                    id: self.ligands_count,
                    position: Array1::from_vec(vec![positions[i * 2], positions[i * 2 + 1]]),
                    velocity: Array1::from_vec(vec![0.0, 0.0]), // velocity is not tracked after collision
                    message: messages[i]
                };
                ligands.push(ligand);
            }
            
            for ligand in ligands {
                println!("Received ligand ID: {}, message: {}, position: {:?}", ligand.id, ligand.message, ligand.position);
            }

            self.ligands = vec![];
            self.ligands_count = 0;

            unsafe {
            // get ligands to host
            let device_ligands = self.cuda_world.as_mut().unwrap().get_ligand_arrays();
            let message_pointer = libc::malloc(device_ligands.num_ligands * std::mem::size_of::<u32>()) as *mut u32;
            let position_pointer = libc::malloc(device_ligands.num_ligands * 2 * std::mem::size_of::<f32>()) as *mut f32;

        
            use crate::cuda::cuda_bindings::memory_gpu as mem;
            mem::copy_DtoH_u(message_pointer, device_ligands.message, device_ligands.num_ligands as u32);
            mem::copy_DtoH_f(position_pointer, device_ligands.pos, device_ligands.num_ligands as u32 * 2);

            let messages_host = std::slice::from_raw_parts(message_pointer, device_ligands.num_ligands);
            let positions_host = std::slice::from_raw_parts(position_pointer, device_ligands.num_ligands * 2);

            for i in 0..device_ligands.num_ligands {
                if messages_host[i] == 0 {
                    continue; // skip empty, collided ligands
                }
                let pos = [positions_host[i * 2], positions_host[i * 2 + 1]];
                
                self.ligands.push(objects::Ligand {
                    id: 0, // id is not important here
                    position: Array1::from_vec(vec![pos[0], pos[1]]),
                    velocity: Array1::from_vec(vec![0.0, 0.0]), // velocity is not tracked after collision
                    message: messages_host[i],
                });
                self.ligands_count += 1;
            }

            libc::free(message_pointer as *mut libc::c_void);
            libc::free(position_pointer as *mut libc::c_void);
            }


        }
    }
}