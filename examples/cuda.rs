#[cfg(feature = "cuda")]
fn main() {
    use UniSim::prelude::*;

    let settings: Settings = settings!("template.json");

    let mut world = World::new(settings);

    edit_settings!(&mut world, store_capacity = 10000, cuda_slots_per_cell = 5);

    world.save(Some("testfiles/test_cuda.bin"), false).expect("Failed to save world");
    world.cuda_initialize().expect("Failed to initialize CUDA");

    let _ = world.add_ligand_source(vec![10.0, 10.0], 10000.0, 1);

    world.run(10000);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Please enable the 'cuda' feature to run this example.");
}
