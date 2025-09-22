#![cfg(all(feature = "debug", feature = "cuda"))]

#![feature(macro_metavar_expr_concat)]


use std::time::{Duration, Instant};
use UniSim::prelude::*;


fn test_performance_cpu(n : usize, ligands: usize) -> Duration {

    let s = settings!(1000, spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (100,100), give_start_vel = true);

    let mut world = World::new(s);

    world.add_ligands(ligands);

    let start = Instant::now();
    world.run(n);
    let duration = start.elapsed();

    return duration;

}

#[cfg(feature = "cuda")]
fn test_performance_gpu(n : usize, ligands: usize) -> Duration {

    let s = settings!(1000, spawn_size = 1.0, fps = 60.0, velocity = 3.0, dimensions = (100,100), give_start_vel = true);
    let mut world = World::new(s);
    world.cuda_initialize().expect("Failed to initialize CUDA");

    world.add_ligands(ligands);

    let start = Instant::now();
    world.run(n);
    let duration = start.elapsed();

    return duration;

}


#[cfg(feature = "cuda")]
fn test_compare_performance() {
    let n = 1000;
    let mut cpu_durations = Vec::new();
    let mut gpu_durations = Vec::new();

    let numbers = [10, 100, 500, 1000, 100000, 1000000];

    for ligands in numbers.iter() {
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
            numbers[i].to_string(),
            cpu_durations[i].as_millis().to_string(),
            gpu_durations[i].as_millis().to_string(),
        ]).expect("Failed to write record");
    }
    wtr.flush().expect("Failed to flush CSV writer");
    println!("Performance results saved to performance_results.csv");

}


fn main(){
    #[cfg(feature = "cuda")]
    test_compare_performance();
}


