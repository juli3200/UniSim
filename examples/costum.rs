use UniSim::prelude::*;

fn main() {
    let runtime = 3000;

    let settings = settings!("template.json");

    let mut world = World::new(settings);

    edit_settings!(&mut world, store_capacity = runtime, fps = 60.0);

    world.save("testfiles/costum.bin", true).expect("failed to save world");
    world.cuda_initialize().expect("");

    world.add_ligand_source(vec![1.0, 10.0], 1000.0, 1, 0.1);

    world.run(runtime);


}