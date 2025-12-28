use UniSim::prelude::*;


fn main() {
    let mut world = World::new(settings!(default_population = 1, spawn_size = 1.0, store_capacity = 100, velocity = 5.0, drag = 0.5, give_start_vel = true, general_force = (0.0, -0.1)));

    world.save(Some("testfiles/gravity_test.bin"), false).expect("Failed to save world");

    world.run(10000);


}