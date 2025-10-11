use UniSim::prelude::*;


fn main() {
    let mut world = World::new(settings!(10, spawn_size = 1.0, store_capacity = 10000, velocity = 5.0, drag = 0.5, give_start_vel = true, gravity = vec![0.0, -0.1]));

    world.save("testfiles/gravity_test.bin").expect("Failed to save world");

    world.run(10000);


}