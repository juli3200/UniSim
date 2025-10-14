#[cfg(feature = "cuda")]
fn main() {
    use UniSim::prelude::*;

    let settings: Settings = settings!("template.json");

    let mut world = World::new(settings);

    edit_settings!(&mut world, store_capacity = 1000, cuda_slots_per_cell = 5);

    world.save("testfiles/test_cuda.bin").expect("Failed to save world");
    world.cuda_initialize().expect("Failed to initialize CUDA");

    world.add_ligand_source(vec![1.0, 1.0], 100.0, 1786, 0.01);
    //world.add_ligands(10);

    world.run(1000);
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Please enable the 'cuda' feature to run this example.");
}
