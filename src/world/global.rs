use crate::world::serialize::{Save, HEADER_SIZE};

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
            iteration: 0,

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

use std::io::{self, Write, Seek, SeekFrom, Read};
use std::fs::{File, OpenOptions};


// save impl Block
impl World{

    fn save_state(&mut self) -> io::Result<()> {
        // Save the current state of the world

        // e.g. self.saved_states = 1024
        // will activate at 1024
        // actual size is 1025 one slot for the next jumper
        if self.saved_states % self.settings.store_capacity == 0 && self.saved_states != 0 {
            // add new capacity
            println!("Increasing save capacity to {}", self.settings.store_capacity * (self.iteration-1));
            println!("Saved states: {}, Population size: {}", self.saved_states, self.population_size);
            self.save_table()?;

        }
        

        let jumper_location = HEADER_SIZE // header
            + self.saved_states * 4 // current jumper 
            + (self.settings.store_capacity +1) * (self.iteration-1); // 4 bytes per entry -> u32

        let jumper_target = self.byte_counter as u32;
        
        // open file and add the jumper coordinate
        let mut file = OpenOptions::new()
            .write(true)
            .open(self.path.as_ref().unwrap())?;
        file.seek(std::io::SeekFrom::Start(jumper_location as u64))?;
        file.write_all(&jumper_target.to_le_bytes())?;

        drop(file); // close the file

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

        let jumper_table_size = 4 * (self.settings.store_capacity + 1); // 4 bytes per entry -> u32 // 1 for pointer at next jumper
        let mut jumper_table = Vec::with_capacity(0);
        jumper_table.resize(jumper_table_size, 0u8);

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(self.path.as_ref().unwrap())?;
        file.write_all(&jumper_table)?;

        self.byte_counter += jumper_table_size;
        self.iteration += 1;

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


    fn find_location(&self, slot: Option<usize>) -> Option<u32> {
        // clause if table was not called

        if self.iteration == 0 {
            return Some(HEADER_SIZE as u32);
        }
        

        fn inversive_find(iteration: u32, saved_states: u32, store_capacity: u32, location: u32, file: &mut File) -> Option<u32> {

            if iteration == 1 {
                return Some(location + saved_states % store_capacity * 4);
            }

            // opening the file at location
            let e1 = file.seek(SeekFrom::Start(location as u64));

            // reading the next location to the buffer
            let mut buffer = [0u8; 4];
            let e2 = file.read_exact(&mut buffer);

            // convert the buffer in a u32
            let next_location = u32::from_le_bytes(buffer);

            // Error handling
            if e1.is_err() || e2.is_err() {
                return None;
            }

            // recursive call
            inversive_find(iteration-1, saved_states, store_capacity, next_location, file)
            
        }

        let iteration;
        let target_slot;

        match slot {
            Some(slot) => {
                iteration = (slot as f32 / self.settings.store_capacity as f32).ceil() as u32;
                target_slot = slot as u32;
            }
            None => {
                iteration = self.iteration as u32;
                target_slot = self.saved_states as u32;
            }
        }

        // set location to default HEADER_SIZE
        let location = HEADER_SIZE as u32;

        // Error handled by returning None in case of failure
        if let Ok(mut file) = File::open(self.path.as_ref().unwrap()) {
            return inversive_find(iteration, target_slot, self.settings.store_capacity as u32, location, &mut file);
        }

        None
    }
} 