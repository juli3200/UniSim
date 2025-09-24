#[cfg(feature = "cuda")]
fn main() {
    use UniSim::prelude::*;

    let settings: Settings = settings!(1, spawn_size = 1.0, give_start_vel = true, velocity = 20.0, fps = 60.0, store_capacity = 10000, dimensions=(100,100));
    let mut world = World::new(settings);

    //world.save("test_cuda.bin").expect("Failed to save world");
    world.cuda_initialize().expect("Failed to initialize CUDA");


    world.run(10000);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Please enable the 'cuda' feature to run this example.");
}
