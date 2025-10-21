#[cfg(feature = "cuda")]
fn main() {
    use UniSim::prelude::*;

    let settings: Settings = settings!("template.json");

    let mut world = World::new(settings);

    edit_settings!(&mut world, store_capacity = 100, cuda_slots_per_cell = 5);

    world.save("testfiles/test_cuda.bin").expect("Failed to save world");
    world.cuda_initialize().expect("Failed to initialize CUDA");

    world.add_ligand_source(vec![10.0, 10.0], 10000.0, 1, 0.0001);

    world.run(100);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Please enable the 'cuda' feature to run this example.");
}
