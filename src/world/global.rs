use super::*;


impl World{
    pub fn new(settings: Settings) -> Self {
        let mut world = Self {
            settings,
            time: 0.0,
            population_size: 0,
            ligands_count: 0,
            counter: 0,

            // the objects are filled in in the initialize function
            entities: Vec::new(),
            ligands: Vec::new(),
            space: Space::empty(),
        };
        world
            .initialize()
            .expect("Failed to initialize world");

        world
    }

    fn initialize(&mut self) -> Result<(), String> {

        // Initialize the space
        self.space = Space::new(self.settings.dimensions)?;

        // Initialize the world with default population size
        for _ in 0..self.settings.default_population {
            let entity = objects::Entity::new(self.counter);
            self.entities.push(entity);
            self.population_size += 1;
            self.counter += 1;
        }




        Ok(())
    }
}