use UniSim::prelude::*;


fn main() {

    let settings: Settings = settings!(100, spawn_size = 1.0, give_start_vel = true, velocity = 3.0, fps = 60.0, store_capacity = 5000, dimensions=(100,100));

    let mut world = World::new(settings);


    //world.add_ligand_source(vec![1.0, 1.0], 10.0, 0);

    #[cfg(feature = "debug")]
    {
        //world.add_ligands(1000);
        
    }
    let e = world.save("testfiles/test.bin");

    if let Err(e) = e {
        eprintln!("Failed to save world: {}", e);
    }

    world.run(5000);
}
