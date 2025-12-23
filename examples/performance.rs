#[cfg(all(not(feature = "cuda"), feature = "debug"))]
mod performance {
    pub fn test_compare_performance() {
        println!("Performance comparison is only available with both 'debug' and 'cuda' features enabled.");

        if !cfg!(feature = "rayon") {
            println!("However, the 'rayon' feature is disabled.");
        } else {
            panic!();
        }
        println!("Testing without rayon feature.");

        let n = 2000;
        let mut cpu_durations = Vec::new();

        let numbers = [100, 1000, 10_000, 100_000];
        let ligand_source_numbers = [1.0, 10.0, 100.0, 1000.0];

        for ligands in 0..numbers.len() {
            println!("Testing with {} ligands and {} Ligands per second emission", numbers[ligands], ligand_source_numbers[ligands]);
            println!("CPU:");
            let cpu_duration = super::test_performance_cpu(n, numbers[ligands], Some(ligand_source_numbers[ligands]));
            cpu_durations.push(cpu_duration);
            println!("CPU Duration: {:?}", cpu_duration);
            println!("-----------------------");
        }
    }
}

use std::time::{Duration, Instant};
use UniSim::prelude::*;

fn test_performance_cpu(n : usize, ligands: usize, ligand_source: Option<f32>) -> Duration {

    let s = settings!(spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (100,100), give_start_vel = true);

    let mut world = World::new(s);

    world.initialize().expect("Failed to initialize world");

    world.add_ligands(ligands);
    if let Some(rate) = ligand_source {
        let _ = world.add_ligand_source(vec![50.0, 50.0], rate, 15);
    }

    let start = Instant::now();
    world.run(n);
    let duration = start.elapsed();

    return duration;

}

#[cfg(all(feature = "debug", feature = "cuda"))]
mod performance {
    use std::time::{Duration, Instant};
    use UniSim::prelude::*;
    use super::test_performance_cpu;


    fn test_performance_gpu(n : usize, ligands: usize, ligand_source: Option<f32>) -> Duration {

        let s = settings!(spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (100,100), give_start_vel = true);
        let mut world = World::new(s);
        world.cuda_initialize().expect("Failed to initialize CUDA");

        world.add_ligands(ligands);
        if let Some(rate) = ligand_source {
            let _ = world.add_ligand_source(vec![50.0, 50.0], rate, 15);
        }

        let start = Instant::now();
        world.run(n);
        let duration = start.elapsed();

        return duration;

    }



    pub fn test_compare_performance() {
        let n = 2000;
        let mut cpu_durations = Vec::new();
        let mut gpu_durations = Vec::new();

        let numbers = [0; 6];
        let ligand_source_numbers = [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0];

        for ligands in 0..numbers.len() {
            println!("Testing with {} ligands and {} Ligands per second emission", numbers[ligands], ligand_source_numbers[ligands]);
            println!("CPU:");
            let cpu_duration = test_performance_cpu(n, numbers[ligands], Some(ligand_source_numbers[ligands]));
            cpu_durations.push(cpu_duration);
            println!("CPU Duration: {:?}", cpu_duration);
            println!("GPU:");
            let gpu_duration = test_performance_gpu(n, numbers[ligands], Some(ligand_source_numbers[ligands]));
            gpu_durations.push(gpu_duration);
            println!("GPU Duration: {:?}", gpu_duration);
            println!("-----------------------");
        }

        // store results in csv file
        let mut wtr = csv::Writer::from_path("performance_results.csv").expect("Failed to create CSV writer");
        wtr.write_record(&["Ligands", "CPU Duration (ms)", "GPU Duration (ms)"]).expect("Failed to write header");
        for i in 0..cpu_durations.len() {
            wtr.write_record(&[
                numbers[i].to_string(),
                cpu_durations[i].as_millis().to_string(),
                gpu_durations[i].as_millis().to_string(),
            ]).expect("Failed to write record");
        }
        wtr.flush().expect("Failed to flush CSV writer");
        println!("Performance results saved to performance_results.csv");

    }



}

#[cfg(not(feature = "debug"))]
mod performance {
    pub fn test_compare_performance() {
        println!("Performance comparison is only available with 'debug' feature enabled.");
    }
}

fn main(){
    performance::test_compare_performance();
}