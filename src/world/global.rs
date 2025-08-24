use crate::world::serialize::Save;

use super::*;


impl World {
    pub fn new() -> Self {
        let mut world = Self {
            settings: Settings::default(),
            path: None,
            time: 0.0,
            population_size: 0,
            ligands_count: 0,
            counter: 0,

            // the objects are filled in in the initialize function
            entities: Vec::new(),
            ligands: Vec::new(),
            space: Space::empty(),

            #[cfg(feature = "cuda")]
            cuda_world: None, // CUDA world is initialized later if GPU is active
        };
        world
            .initialize()
            .expect("Failed to initialize world");

        world
    }

    fn initialize(&mut self) -> Result<(), String> {

        // Initialize the space
        self.space = Space::new(&self.settings)?;

        // Initialize the world with default population size
        for _ in 0..self.settings.default_population {
            let entity = objects::Entity::new(self.counter, &mut self.space, &self.entities, self.settings.spawn_size)?;
            self.entities.push(entity);
            self.counter += 1;
        }

        self.population_size = self.settings.default_population;




        Ok(())
    }


    fn update(&mut self){
        // clone entities because of borrowing rules
        // and also to avoid double collision resolution (because collision would already be resolved for the original entity)
        let temp_entities = self.entities.clone();

        for entity in &mut self.entities {
            // giving each entity the entities as they where to avoid double resolution
            entity.update(&mut self.space, &temp_entities);
        }
    }


}

use std::io::{self, Write};
use std::fs::OpenOptions;


// save impl Block
impl World{

    fn save_state(&self) -> io::Result<()> {
        // Save the current state of the world

        match self.serialize() {
            Ok(state) => {
                let mut file = OpenOptions::new()
                    .append(true)
                    .open(self.path.as_ref().unwrap())?;
                file.write_all(&state)?;
            }
            Err(e) => {
                return Err(io::Error::new(io::ErrorKind::Other, format!("Failed to serialize state: {}", e)));
            }
        }

        Ok(())
    }

    fn save_header(&self) -> io::Result<()> {
        match serialize::serialize_header(self) {
            Ok(header) => {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .open(self.path.as_ref().unwrap())?;
                file.write_all(&header)?;
            }
            Err(e) => {
                return Err(io::Error::new(io::ErrorKind::Other, format!("Failed to serialize header: {}", e)));
            }
        };

        
        Ok(())
    }

    // to be accessed by user
    // update the path where the world is saved
    pub fn save<S>(&mut self, path: S) -> io::Result<()>
    where
        S: AsRef<std::path::Path>,
    {
        self.path = Some(path.as_ref().to_path_buf());
        self.save_header()?;
        Ok(())
    }

    pub fn stop_save(&mut self) -> Result<(), String> {
        self.path = None;
        Ok(())
    }


}