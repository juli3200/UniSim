use UniSim::prelude::*;

fn main() {

    let settings = settings!("template.json");

    let mut world = World::new(settings);

    world.run(1000);


}