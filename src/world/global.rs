use crate::world::serialize::Save;
use libc;
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
        for _ in 0..self.settings.default_population() {
            let entity = objects::Entity::new(self.counter, &mut self.space, &self.entities, &self.settings)?;
            self.entities.push(entity);
            self.counter += 1;
        }

        self.population_size = self.settings.default_population();




        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_initialize(&mut self) -> Result<(), String> {
        // Initialize the CUDA world and activate GPU processing
        if self.cuda_world.is_some() {
            return Err("CUDA world is already initialized".to_string());
        }

        // test if gpu is available
        if unsafe{crate::cuda::cuda_bindings::tests_gpu::cuda_test() != 0} {
            return Err("CUDA is not available on this system".to_string());
        }
        
        let cuda_world = crate::cuda::CUDAWorld::new(&self.settings, &self.entities, &self.ligands);
        self.cuda_world = Some(cuda_world);

        Ok(())
    }

    pub fn run(&mut self, n: usize) {
        // Main loop for the world simulation
        for _ in 0..n {
            self.update();
        }
    }

    pub(crate) fn update(&mut self){

        #[cfg(feature = "cuda")]
        {
        if self.cuda_world.is_some() {
            match self.gpu_update() {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("GPU update failed: {}, switching to CPU", e);
                    self.cuda_world = None; // disable GPU processing on error
                    self.cpu_update();
                }
            }
        } else {
            self.cpu_update();
        }
        }

        #[cfg(not(feature = "cuda"))]
        {
            self.cpu_update();
        }

        
        self.time += 1.0 / self.settings.fps() as f32;

                // exit if saving is disabled
        if self.path.is_none(){return;}

        match self.serialize() {
            // serialize the state
            Ok(state) => {
                // add it to the buffer
                self.buffer.push(state);

                if self.buffer.len() == self.settings.store_capacity() {

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

    fn cpu_update(&mut self) {
        // Update the world using CPU processing
        // update all entities positions
        for entity in &mut self.entities {
            entity.update(&mut self.space);
        }

        let entities_clone = self.entities.clone();

        // check for collisions
        for i in 0..self.entities.len() {
            self.entities[i].resolve_collision(&mut self.space, &entities_clone);
        }

    }

    #[cfg(feature = "cuda")]
    fn gpu_update(&mut self) -> Result<(), String> {


        if self.cuda_world.is_none() {
            return Err("CUDA world is not initialized".to_string());
        }
        println!("GPU update");

        // Update the world using GPU processing

        // entities are updated on CPU
        // update all entities positions and receive Ligands

        let mut new_ligands = Vec::new();

        for entity in  self.entities.iter_mut() {
            entity.update(&mut self.space);
            new_ligands.extend(entity.emit_ligands()); // not yet implemented
        }

        let entities_clone = self.entities.clone();

        // check for collisions
        for i in 0..self.entities.len() {
            self.entities[i].resolve_collision(&mut self.space, &entities_clone);
        }

        // ligands are updated on GPU
        if cfg!(test) {
            new_ligands = self.ligands.clone(); // in test mode, use the ligands from the world, so ligands can be added manually
        }

        // please improve this code
        let mut ligands_pos: Vec<f32> = new_ligands.iter()
            .flat_map(|l| l.position.iter())
            .cloned()
            .collect();
        let mut ligands_vel: Vec<f32> = new_ligands.iter()
            .flat_map(|l| l.velocity.iter())
            .cloned()
            .collect();
        let mut ligands_content: Vec<u32> = new_ligands.iter()
            .map(|l| l.message) 
            .collect();

        let err = self.cuda_world.as_mut().unwrap().add_ligands(&mut ligands_pos, &mut ligands_vel, &mut ligands_content);

        
        if let Err(e) = err {
            if e == -1 {
                return Err("Input ligand vectors have incorrect sizes".to_string());
            } else {
                // increase capacity 
                self.cuda_world.as_mut().unwrap().increase_cap(objects::ObjectType::Ligand(0));
            }
        }

        // get the received ligands from the entities
        let received_ligands = self.cuda_world.as_mut().unwrap().update(self.space.max_size.ceil() as u32);

        let len = received_ligands.counter as usize;

        // slice around the *mut pointers
        let messages: &[u32];
        let positions: &[f32];
        let ids: &[u32];

        unsafe {
            messages = std::slice::from_raw_parts(received_ligands.collided_message, len);
            positions = std::slice::from_raw_parts(received_ligands.collided_pos, len * 2);
            ids = std::slice::from_raw_parts(received_ligands.collided_entities, len);
        }

        // add the ligands to entities and edit concentrations
        for i in 0..len {
            use crate::world::info::get_entity_mut;

            let pos = [positions[i * 2], positions[i * 2 + 1]];
            let message = messages[i];
            let entity_id = ids[i] as usize;

            // find the entity with the corresponding id
            let entity_ref = get_entity_mut(&mut self.entities, entity_id);

            if let Some(entity) = entity_ref {
                entity.receive_ligand(message, pos)?;
            } else {
                return Err(format!("Entity with ID {} not found", entity_id));
            }
        }


        // DEBUGGING
        #[cfg(test)]
        self.copy_ligands(positions, messages, len);

        // free the collision arrays
        unsafe {
            libc::free(received_ligands.collided_message as *mut libc::c_void);
            libc::free(received_ligands.collided_pos as *mut libc::c_void);
            libc::free(received_ligands.collided_entities as *mut libc::c_void);
        }



        

        Ok(())
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
        let mut bytes_written = self.byte_counter + (self.settings.store_capacity()+ 1) * 4; // add space for all the jumpers and the jumper for the next jumper

        let mut jumper_table = Vec::with_capacity((self.settings.store_capacity() + 1)* 4); // 4 bytes per entry -> u32 + 4 bytes for next jumper table

        // fill the jumper_table with the addresses for the jumper
        for i in 0..self.settings.store_capacity(){
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