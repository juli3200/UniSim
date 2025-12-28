#[cfg(feature = "cuda")]
fn main() {
    use UniSim::prelude::*;
    for i in 1..5 {

        let settings: Settings = settings!("experiments/mutation.json");

        let mut world = World::new(settings);

        edit_settings!(&mut world, mutation_rate = 0.01 * i as f64, path = format!("testfiles/mutation_rate_{}.bin", i));

        world.save(None, true).expect("Failed to save world");
        world.cuda_initialize().expect("Failed to initialize CUDA");

        let _ = world.add_ligand_source(vec![10.0, 10.0], 10000.0, 1);

        world.run(10000);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Please enable the 'cuda' feature to run this example.");
}
