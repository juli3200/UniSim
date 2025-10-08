use UniSim::prelude::*;


fn main() {

    let settings: Settings = settings!(100, spawn_size = 1.0, give_start_vel = true, velocity = 20.0, fps = 60.0, store_capacity = 1000, dimensions=(100,100));

    let mut world = World::new(settings);

    edit_settings!(&mut world, fps = 60.0, velocity = 3.0, gravity = vec![0.0, -1.0]);

    #[cfg(feature = "debug")]
    world.add_ligands(1000);

    let e = world.save("testfiles/test.bin");

    if let Err(e) = e {
        eprintln!("Failed to save world: {}", e);
    }

    world.run(1000);
}
