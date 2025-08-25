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
            byte_counter: 0,
            saved_states: 0,

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

    fn save_state(&mut self) -> io::Result<()> {
        // Save the current state of the world

        let len;

        match self.serialize() {
            Ok(state) => {
                len = state.len();
                let mut file = OpenOptions::new()
                    .append(true)
                    .open(self.path.as_ref().unwrap())?;
                file.write_all(&state)?;
                
            }
            Err(e) => {
                return Err(io::Error::new(io::ErrorKind::Other, format!("Failed to serialize state: {}", e)));
            }
        }

        self.byte_counter += len;
        self.saved_states += 1;

        Ok(())
    }

    fn save_header(&mut self) -> io::Result<()> {

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

        self.byte_counter += serialize::HEADER_SIZE;

        Ok(())
    }

    fn save_table(&mut self) -> io::Result<()> {
        // allocates capacity for the jumper table to the file

        let jumper_table_size = 4 * self.settings.store_capacity; // 4 bytes per entry -> u32
        let mut jumper_table = Vec::with_capacity(0);
        jumper_table.resize(jumper_table_size, 0u8);

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(self.path.as_ref().unwrap())?;
        file.write_all(&jumper_table)?;

        self.byte_counter += jumper_table_size;
        
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
        self.save_table()?;
        Ok(())
    }

    pub fn stop_save(&mut self) -> Result<(), String> {
        self.path = None;
        Ok(())
    }


}