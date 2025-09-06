#![feature(macro_metavar_expr_concat)]

use UniSim::world;
use UniSim::Settings;
use UniSim::settings;





fn main() {

    let settings: UniSim::Settings = settings!(100, spawn_size = 1.0, give_start_vel = true, velocity = 20.0, fps = 60.0, store_capacity = 100000, dimensions=(100,100));
    let mut world = world::World::new(settings);

    let e = world.save("test.bin");

    if let Err(e) = e {
        eprintln!("Failed to save world: {}", e);
    }

    world.run(100000);
}
