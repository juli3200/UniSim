use UniSim::prelude::*;

fn main() {
    let runtime = 3000;

    let settings = settings!("template.json");

    let mut world = World::new(settings);

    edit_settings!(&mut world, store_capacity = runtime*4, fps = 60.0);

    world.save(Some("testfiles/4_filters.bin"), true).expect("failed to save world");
    world.cuda_initialize().expect("");

    for i in 1..5 {
        let _ = world.add_ligand_source(vec![50.0, 50.0], 10000.0, i);
        world.run(runtime);
        world.delete_all_ligands();
    }
}