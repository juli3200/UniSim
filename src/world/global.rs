use crate::world::serialize::Save;
use crate::world::info::{get_entity, get_entity_mut};
use crate::prelude::*;
use rayon::prelude::*;
use super::*;

#[cfg(feature = "cuda")]
use libc;


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
            counter: 1, // start counting entities from 1 (0 is reserved for empty) (cuda)
            byte_counter: 0,
            iteration: 0,

            // the objects are filled in in the initialize function
            entities: Vec::new(),
            ligands: Vec::new(),
            ligand_sources: Vec::new(),
            new_ligands: Vec::new(),
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
            let entity = objects::Entity::new(self.counter, &mut self.space, &self.entities, &self.settings, None)?;
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

    pub fn add_ligand_source<A: Into<Array1<f32>>>(&mut self, position: A, emission_rate: f32, ligand_spec: u16, ligand_energy: f32) {
        let source = objects::LigandSource::new(position.into(), emission_rate, ligand_spec, ligand_energy);
        self.ligand_sources.push(source);
    }

    pub fn run(&mut self, n: usize) {

        // check if if n is smaller than store capacity$
        // it isn't saved if too small
        if !self.settings.is_init(){
            if n + self.buffer.len() < self.settings.store_capacity() {
                eprint!("Warning: Number of steps to run is smaller than store capacity, state will not be saved.");
                if self.iteration == 0 {
                    eprintln!("Do you want to decrease the store capacity? (y/n)");
                    let mut input = String::new();   
                    std::io::stdin().read_line(&mut input).expect("Failed to read line");
                    if input.trim() == "y" {      
                        use crate::edit_settings;
                        edit_settings!(self, store_capacity = n + self.buffer.len());
                        eprintln!("Store capacity set to {}", self.settings.store_capacity());
                    } else {
                        eprintln!("Continuing without saving.");
                    }
                }           
            }

            self.settings.init();
        }

        // Main loop for the world simulation
        for i in 0..n {
            self.update();
            if i % 100 == 0 {
                println!("Step {}/{}", i, n);
                #[cfg(feature = "debug")]
                {   
                    #[cfg(feature = "cuda")]
                    if let Some(cuda_world) = &self.cuda_world {

                        println!("Entities: {}, Ligands: {}, CUDA active", self.entities.len(), cuda_world.ligand_count);
                    }
                    println!("Entities: {}, Ligands: {}", self.entities.len(), self.ligands.len());
                }
            }
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
        // add new ligands to the world
        self.ligands.extend(self.new_ligands.drain(..));
        self.ligands_count = self.ligands.len();
        // clear the new ligands vector
        self.new_ligands.clear();


        // Update the world using CPU processing
        // update all entities positions
        for entity in &mut self.entities {
            entity.update_physics(&mut self.space);
        }

        let entities_clone = self.entities.clone();

        // check for collisions
        for i in 0..self.entities.len() {
            self.entities[i].resolve_collision(&mut self.space, &entities_clone);
        }


        // update all ligands positions and check for collisions
        let mut collided = vec![None; self.ligands.len()];

        let dt = 1.0 / self.settings.fps() as f32;

        for (i, ligand) in self.ligands.iter_mut().enumerate() {
            if let Some(entity_id) = ligand.update(&self.space, &entities_clone, dt) {
                collided[i] = Some(entity_id);
            }
        }

        // remove collided ligands and add them to entities

        let len = self.ligands.len();

        for i in (0..len).rev() {
            if let Some(entity_id) = collided[i] {
                let entity_ref = get_entity_mut(&mut self.entities, entity_id);

                if let Some(entity) = entity_ref {
                    let remove = entity.receive_ligand(&self.ligands[i], &self.settings);
                    // remove the ligand if it was absorbed else turn it around
                    if remove {
                        self.ligands.remove(i);
                    } else {
                        // re-emit the ligand
                        self.ligands[i].re_emit();
                    }
                } else {
                    eprintln!("Entity with ID {} not found", entity_id);
                }
            }
        }

        for entity in &mut self.entities {
            entity.update_output(&self.settings);
        }


        // emit new ligands from entities
        for entity in &mut self.entities {
            let new_ligands = entity.emit_ligands();
            self.new_ligands.extend(new_ligands);
        }

        // emit new ligands from sources
        for source in &self.ligand_sources {
            let new_ligands = source.emit_ligands(dt);
            self.new_ligands.extend(new_ligands);

        }

        self.ligands_count = self.ligands.len();

        

    }

    #[cfg(feature = "cuda")]
    fn gpu_update(&mut self) -> Result<(), String> {

        if self.cuda_world.is_none() {
            return Err("CUDA world is not initialized".to_string());
        }

        // Update the world using GPU processing

        // entities are updated on CPU
        // update all entities positions and receive Ligands

        for entity in  self.entities.iter_mut() {
            entity.update_physics(&mut self.space);
        }

        let entities_clone = self.entities.clone();

        // check for collisions on CPU
        for i in 0..self.entities.len() {
            self.entities[i].resolve_collision(&mut self.space, &entities_clone);
        }

        // add new ligands to the cuda world
        let err = self.cuda_world.as_mut().unwrap().add_ligands(&self.new_ligands);

        self.new_ligands.clear();

        // error handling for adding ligands
        if let Err(_) = err {
            // increase capacity 
            use crate::cuda;
            println!("Increasing ligand capacity");
            self.cuda_world.as_mut().unwrap().increase_cap(cuda::IncreaseType::Ligand);
        }

        // get the received ligands from the entities
        let (received_ligands, overflow) = self.cuda_world.as_mut().unwrap().update(&self.entities, self.space.max_size.ceil() as u32);

        if overflow > 0 {
            use crate::{cuda, edit_settings};

            println!("Warning: Grid overflow occurred, increasing grid size or slots per cell");
            let new_size = (self.settings.cuda_slots_per_cell() as f32 * 1.2) as usize;

            // edit the settings to increase the grid size
            edit_settings!(self, cuda_slots_per_cell = new_size);

            // recreate the grid with the new size
            self.cuda_world.as_mut().unwrap().increase_cap(cuda::IncreaseType::Grid);
        }


        let len = received_ligands.count as usize;
        dbg!(len);
        dbg!(overflow);
        dbg!(received_ligands.energies.is_null());

        // check if the pointers are null
        if !received_ligands.receptor_ids.is_null() & !received_ligands.energies.is_null() {
            // slice around the *mut pointers
            let energies: &[f32];
            let receptors: &[u32];

            unsafe {
                energies = std::slice::from_raw_parts(received_ligands.energies, len);
                receptors = std::slice::from_raw_parts(received_ligands.receptor_ids, len * 2);
            }

            // add the ligands to entities and edit concentrations
            for i in 0..len {
                let entity_id = (receptors[i] as f32 / self.settings.receptor_capacity() as f32).floor() as usize;
                let receptor_index = (receptors[i] % self.settings.receptor_capacity() as u32) as usize;
                let energy = energies[i];
        
                // find the entity with the corresponding id
                let entity_ref = get_entity_mut(&mut self.entities, entity_id);

                if let Some(entity) = entity_ref {
                    // can go through the shortcut because the bond was already checked on the GPU
                    entity.receive_ligand_cuda_shortcut(energy, receptor_index, &self.settings);
                }
            }

        } else {
            eprintln!("Received null pointer from CUDA world");
        }

        // if save ligands feature is active copy ligands from gpu to cpu
        #[cfg(feature = "save_ligands")]
        {
            self.copy_ligands();
        }


        // free the collision arrays
        unsafe {
            libc::free(received_ligands.receptor_ids as *mut libc::c_void);
            libc::free(received_ligands.energies as *mut libc::c_void);
        }

        let dt = 1.0 / self.settings.fps() as f32;
        for entity in &mut self.entities {
            entity.update_output(&self.settings);
        }

        // emit new ligands from entities
        for entity in &mut self.entities {
            let new_ligands = entity.emit_ligands();
            self.new_ligands.extend(new_ligands);
        }

        // emit new ligands from sources
        for source in &self.ligand_sources {
            let new_ligands = source.emit_ligands(dt);
            self.new_ligands.extend(new_ligands);

        }

        Ok(())
    }


    pub fn print_entity_stats(&self, entity_id: usize) {
        // Print statistics of a specific entity by its ID
        if let Some(entity) = get_entity(&self.entities, entity_id){
            entity.print_stats();
        } else {
            println!("Entity with ID {} not found", entity_id);
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
        // check if the path exists
        if path.as_ref().exists() {
            eprint!("Warning: File {} already exists. Overwrite? (y/n)", path.as_ref().display());
            let mut input = String::new();   
            std::io::stdin().read_line(&mut input).expect("Failed to read line");
            if input.trim() != "y" {      
                return Err(io::Error::new(io::ErrorKind::Other, "File already exists"));
            }
            std::fs::remove_file(path.as_ref())?;
        }


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

// debug features

#[cfg(feature = "debug")]
impl World {
    
    // add n ligands at random positions
    // only for testing purposes
    pub fn add_ligands(&mut self, n: usize) {
        use rand::Rng;

        let mut rng = rand::rng();

        for _ in 0..n {
            let x = rng.random_range(0.0..self.space.width as f32);
            let y = rng.random_range(0.0..self.space.height as f32);
            let position = Array1::from_vec(vec![x, y]);


            let len = (position[0].powi(2) + position[1].powi(2)).sqrt();
            let norm_pos = Array1::from_vec(vec![position[0]/len, position[1]/len]);
            // add ligand at random position
            // ensure position is within bounds
            let ligand = objects::Ligand::new(usize::MAX, 0.2, 0u16, position, norm_pos);
            self.ligands.push(ligand);
            self.ligands_count += 1;
        }
    }

    pub fn change_concentration(&mut self, index: usize, value: i16)  {
        for i in 0..self.entities.len() {
        
            if index < self.entities[i].inner_protein_levels.len() {
                self.entities[i].inner_protein_levels[index] = value;
            } else {
                eprint!("Index {} out of bounds for concentrations", index);
            }
            return;
            
        }
    }

}
#[cfg(feature = "save_ligands")]
// #[cfg(feature = "save_ligands")]
// Test debugging impl block for World
#[cfg(feature = "cuda")]
impl World{

    pub(crate) fn copy_ligands(&mut self){
        use crate::cuda;

        self.ligands.clear(); // daaaaaaaaaaaa

        if self.cuda_world.is_none() {
        return;
        }

        let cuda_world = self.cuda_world.as_mut().unwrap();

        let ligands_h = unsafe { libc::malloc(cuda_world.ligand_count as usize * std::mem::size_of::<cuda::LigandCuda>()) as *mut cuda::LigandCuda };
          
        use cuda::cuda_bindings::memory_gpu as cu_mem;

        unsafe{cu_mem::copy_DtoH_ligand(ligands_h, cuda_world.ligands, cuda_world.ligand_count);}

        if ligands_h.is_null() {
            eprintln!("Failed to allocate memory for ligands copy");
            return;
        }

        let ligands_slice = unsafe { std::slice::from_raw_parts(ligands_h, cuda_world.ligand_count as usize) };

        for i in 0..cuda_world.ligand_count as usize {
            let ligand_cuda = &ligands_slice[i];
            if let Ok(ligand) = ligand_cuda.try_into() {
                self.ligands.push(ligand);
            }
        }
    }
}
