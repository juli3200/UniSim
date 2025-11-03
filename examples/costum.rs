use UniSim::prelude::*;

fn main() {

    let settings = settings!("template.json");

    let mut world = World::new(settings);


    world.save("testfiles/costum.bin").expect("failed to save world");
    world.cuda_initialize().expect("");

    world.add_ligand_source(vec![1.0, 10.], 10.0, 1, 0.001);

    world.run(1024);


}