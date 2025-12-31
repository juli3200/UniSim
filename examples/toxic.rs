use UniSim::prelude::*;

fn main() {

    let settings: Settings = settings!("experiments/toxic/toxic.json");


    let mut world = World::new(settings);

    let name = "1";

    world.save(Some(&format!("experiments/toxic/cache/{}", name)), true).expect("Failed to save world");

    world.cuda_initialize().expect("Failed to initialize CUDA");

    // food source
    world.add_ligand_source(vec![50.0, 50.0], 10000.0, 4).expect("Failed to add ligand source");

    // toxin source
    world.add_ligand_source(vec![60.0, 60.0], 10000.0, 1).expect("Failed to add ligand source");

    world.run(5000);

}