use UniSim::prelude::*;

fn main() {

    let settings = settings!("template.json");

    let mut world = World::new(settings);

    world.save("testfiles/costum.bin").expect("failed to save world");

    world.run(1000);


}