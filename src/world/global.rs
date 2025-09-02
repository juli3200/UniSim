use crate::world::serialize::Save;

use super::*;


impl World {

    // creates a new World with n = 100
    pub fn default() -> Self {
        let settings = Settings::new(100);
        Self::new(settings)
    }

    // creates and initializes the World with specified settings
    pub fn new(settings: Settings) -> Self {

        let buffer = Vec::with_capacity(settings.store_capacity() * 1024); // 1 MB buffer

        let mut world = Self {
            settings: settings,
            buffer: buffer,
            path: None,
            time: 0.0,
            population_size: 0,
            ligands_count: 0,
            counter: 0,
            byte_counter: 0,
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
            let entity = objects::Entity::new(self.counter, &mut self.space, &self.entities, &self.settings)?;
            self.entities.push(entity);
            self.counter += 1;
        }

        self.population_size = self.settings.default_population;




        Ok(())
    }

    pub fn run(&mut self, n: usize) {
        // Main loop for the world simulation
        for _ in 0..n {
            self.update();
        }
    }

    pub(crate) fn update(&mut self){
        // clone entities because of borrowing rules
        // and also to avoid double collision resolution (because collision would already be resolved for the original entity)
        let temp_entities = self.entities.clone();

        for entity in &mut self.entities {
            // giving each entity the entities as they where to avoid double resolution
            entity.update(&mut self.space, &temp_entities);
        }

        self.time += 1.0 / self.settings.fps as f32;

        // exit if saving is disabled
        if self.path.is_none(){return;}

        match self.serialize() {
            // serialize the state
            Ok(state) => {
                // add it to the buffer
                self.buffer.push(state);

                if self.buffer.len() == self.settings.store_capacity {

                    // save the state if the buffer is full
                    match self.save_buffer() {
                        Ok(_) => {
                            // clear the buffer
                            self.buffer.clear();
                        }
                        Err(e) => {
                            eprintln!("Failed to save state: {}", e);
                            self.buffer.clear();
                        }
                    }
                }


            }
            Err(e) => {
                eprintln!("Failed to serialize state: {}", e);
            }
        }


        
    }


}

use std::io::{self, Write};
use std::fs::OpenOptions;

// save impl Block
impl World{

    
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

    fn save_buffer(&mut self) -> io::Result<()> {
        // Save the current buffer, containing serialized states of the world

        println!("Writing {} states to disk, iteration {}", self.buffer.len(), self.iteration);

        // save a new jumper table
        self.save_table()?;

        let flat_buffer = self.buffer.iter().flatten().cloned().collect::<Vec<u8>>();

        // write the flat buffer to disk
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(self.path.as_ref().unwrap())?;
        file.write_all(&flat_buffer)?;


        println!("State saved successfully");

        Ok(())
    }


    fn save_table(&mut self) -> io::Result<()> {
        // allocates capacity for the jumper table to the file

        // copy the bytes written to this number for the jumper locations
        let mut bytes_written = self.byte_counter + (self.settings.store_capacity+ 1) * 4; // add space for all the jumpers and the jumper for the next jumper

        let mut jumper_table = Vec::with_capacity((self.settings.store_capacity + 1)* 4); // 4 bytes per entry -> u32 + 4 bytes for next jumper table

        // fill the jumper_table with the addresses for the jumper
        for i in 0..self.settings.store_capacity{
            let state_size = self.buffer[i].len();
            jumper_table.extend((bytes_written as u32).to_le_bytes());
            bytes_written += state_size;
        }

        // jumper to the next jumper table
        jumper_table.extend((bytes_written as u32).to_le_bytes());


        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(self.path.as_ref().unwrap())?;
        file.write_all(&jumper_table)?;

        // Assert that the bytes_written matches the expected file length after writing the jumper table
        let file_len = file.metadata()?.len();
        assert_eq!(bytes_written as u64, file_len, "File length mismatch: expected {}, got {}", bytes_written, file_len);
        self.iteration += 1;

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

} 