
use crate::world::serialize::Save;
use crate::world::info::{get_entity, get_entity_mut};
use crate::prelude::*;
use super::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use libc;


impl World {

    // creates a new World with n = 100
    pub fn default() -> Self {
        let settings = Settings::new();
        Self::new(settings)
    }

    // creates and initializes the World with specified settings
    pub fn new(settings: Settings) -> Self {

        let buffer = Vec::with_capacity(settings.store_capacity() * 2048); // 2 MB buffer

        let world = Self {
            settings: settings,
            buffer: buffer,
            path: None,
            save_genome: false,
            init: false,
            time: 0.0,
            counter: 1, // start counting entities from 1 (0 is reserved for LigandSources)
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
    }

    pub fn initialize(&mut self) -> Result<(), String> {
        if self.init {
            return Err("World already initialized".to_string());
        }

        // Initialize the space
        self.space = Space::new(&self.settings)?;

        // Initialize the world with default population size
        for _ in 0..self.settings.default_population() {
            let entity = objects::Entity::new(self.counter, &mut self.space, &self.entities, &self.settings)?;
            self.entities.push(entity);
            self.counter += 1;
        }

        self.init = true;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_initialize(&mut self) -> Result<(), String> {
        if !self.init {
            self.initialize()?;
        }
        // Initialize the CUDA world and activate GPU processing
        if self.cuda_world.is_some() {
            return Err("CUDA world is already initialized".to_string());
        }

        // test if gpu is available
        if unsafe{crate::cuda::cuda_bindings::tests_gpu::cuda_test() != 0} {
            return Err("CUDA is not available on this system".to_string());
        }

        for i in  0..self.entities.len() {
            self.entities[i].cuda_receptor_index = Some(i as u32);
        }
        
        let cuda_world = crate::cuda::CUDAWorld::new(&self.settings, &self.entities, &self.ligands);
        self.cuda_world = Some(cuda_world);

        Ok(())
    }

    pub fn add_ligand_source<A: Into<Array1<f32>>>(&mut self, position: A, emission_rate: f32, ligand_spec: u16) -> Result<(), String> {
        if self.settings.possible_ligands() <= ligand_spec as usize {
            return Err(format!("Ligand spec {} is out of bounds, maximum is {}", ligand_spec, self.settings.possible_ligands() - 1));
        }
        let source = objects::LigandSource::new(position.into(), emission_rate, ligand_spec);
        self.ligand_sources.push(source);
        Ok(())
    }

    pub fn remove_ligand_source(&mut self, index: usize) -> Result<(), String> {
        if index >= self.ligand_sources.len() {
            return Err(format!("Ligand source index {} is out of bounds, maximum is {}", index, self.ligand_sources.len() - 1));
        }
        self.ligand_sources.remove(index);
        Ok(())
    }

    pub fn remove_all_ligand_sources(&mut self) {
        self.ligand_sources.clear();
    }

    pub fn close(&mut self) {
        // save and close the world
        if self.path.is_some() {
            self.pause_save().unwrap_or_else(|e| {
                eprintln!("Failed to save pause state: {}", e);
            });
            #[cfg(feature = "cuda")]
            if self.cuda_world.is_some() {
                self.cuda_world.as_mut().unwrap().free();
            }
        }
    }

    pub fn run(&mut self, n: usize) {

        if !self.init {
            match self.initialize() {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Failed to initialize world: {}", e);
                    return;
                }
            }
        }

        // check if if n is smaller than store capacity$
        // it isn't saved if too small
        if !self.settings.init(){
            if n + self.buffer.len() < self.settings.store_capacity() {
                if self.path.is_some() {
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
                    #[allow(unused_mut)]
                    let mut no_cuda = true;

                    #[cfg(feature = "cuda")]
                    if let Some(cuda_world) = &self.cuda_world {
                        no_cuda = false;
                        println!("Entities: {}, Ligands: {}, CUDA active", self.entities.len(), cuda_world.ligand_count);
                    }
                    if no_cuda {
                        println!("Entities: {}, Ligands: {}", self.entities.len(), self.ligands.len());
                    }
                }
            }

            // exit early if there are no entities left
            if self.entities.is_empty() {
                // println!("All entities have died, ending simulation at step {}", i);
                // break;
            }

            // rearrange cuda arrays every cuda_memory_interval steps
            // this frees up memory from deleted ligands/entities
            /*
            #[cfg(feature = "cuda")]
            if (i % self.settings.cuda_memory_interval() == 0) && self.cuda_world.is_some() && i != 0{

                self.copy_ligands();

                self.cuda_world.as_mut().unwrap().free();

                self.cuda_world = Some(crate::cuda::CUDAWorld::new(&self.settings, &self.entities, &self.ligands));


                println!("Rearranged CUDA arrays to free memory");
                println!("Entities: {}, Ligands: {}", self.entities.len(), self.ligands.len());
            }
            */
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

        match self.serialize(self.save_genome) {
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

        // clear the new ligands vector
        self.new_ligands.clear();

        let old_positions: Vec<Array1<f32>>;

        // Update the world using CPU processing
        #[cfg(feature = "rayon")]
        {
            // update all entities positions in parallel
            old_positions = self.entities.par_iter_mut().map(|entity| {
                entity.update_physics(&self.space)
            }).collect();
        }

        #[cfg(not(feature = "rayon"))]
        {
            // update all entities positions
            old_positions = self.entities.iter_mut().map(|entity| {
                entity.update_physics(&self.space)
            }).collect();
        }

        // update the positions of all entities in the space grid
        for (i, old_pos) in old_positions.iter().enumerate() {
            self.space.update_entity_position(self.entities[i].id, old_pos.clone(), self.entities[i].position.clone());
        }

        let entities_clone = self.entities.clone();

        // check for collisions
        for i in 0..self.entities.len() {
            self.entities[i].resolve_collision(&mut self.space, &entities_clone);
        }


        // update all ligands positions and check for collisions
        #[allow(unused_assignments)]
        let mut collided = vec![None; self.ligands.len()];

        let dt = 1.0 / self.settings.fps() as f32;

        #[cfg(feature = "rayon")]
        {
            let mut cloned_ligands = self.ligands.clone();
            // update all ligands positions in parallel
            collided = cloned_ligands.par_iter_mut().map(|ligand| 
                ligand.update(&self.space, &entities_clone, dt)).collect();
        }

        #[cfg(not(feature = "rayon"))]
        {
            // update all ligands positions
            for (i, ligand) in self.ligands.iter_mut().enumerate() {
                if let Some(entity_id) = ligand.update(&self.space, &entities_clone, dt) {
                    collided[i] = Some(entity_id);
                }
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
                        self.ligands[i].re_emit(entity , dt);
                    }
                } else {
                    eprintln!("Entity with ID {} not found", entity_id);
                }
            }
        }

        #[cfg(feature = "rayon")]
        {
            // update output concentrations in parallel
            self.entities.par_iter_mut().for_each(|entity| {
                entity.update_output(&self.settings);
            });
        }

        #[cfg(not(feature = "rayon"))]
        {
            // update output concentrations
            for entity in &mut self.entities {
                entity.update_output(&self.settings);
            }
        }

        // emit new ligands from entities
        for entity in &mut self.entities {
            let new_ligands = entity.emit_ligands(&self.settings);
            self.new_ligands.extend(new_ligands);
        }

        // emit new ligands from sources
        for source in &self.ligand_sources {
            let new_ligands = source.emit_ligands(dt, &self.settings);
            self.new_ligands.extend(new_ligands);
        }

        let entities_clone = self.entities.clone();

        // emit new entities
        for entity in 0..self.entities.len() {
            let new_entity = self.entities[entity].reproduce(self.counter, &mut self.space, &entities_clone, &self.settings);
            if let Some(e) = new_entity {
                // add the entity to the world
                self.entities.push(e);
                self.counter += 1;
            }        
        }

        // remove dead entities
        self.entities.retain(|entity| entity.energy > 0.0);
        self.entities.retain(|entity| entity.age as f32 * dt < self.settings.max_age() as f32 || self.settings.max_age() == 0);

    }

    #[cfg(feature = "cuda")]
    fn gpu_update(&mut self) -> Result<(), String> {
        
        // rayon is always enabled with cuda feature

        if self.cuda_world.is_none() {
            return Err("CUDA world is not initialized".to_string());
        }

        // Update the world using GPU processing

        // entities are updated on CPU
        // update all entities positions in parallel
        let old_positions: Vec<Array1<f32>> = self.entities.par_iter_mut().map(|entity| {
            entity.update_physics(&self.space)
        }).collect();

        // update the positions of all entities in the space grid
        for (i, old_pos) in old_positions.iter().enumerate() {
            self.space.update_entity_position(self.entities[i].id, old_pos.clone(), self.entities[i].position.clone());
        }

        let entities_clone = self.entities.clone();

        // check for collisions on CPU
        for i in 0..self.entities.len() {
            self.entities[i].resolve_collision(&mut self.space, &entities_clone);
        }

        // LIGANDS ARE UPDATED ON GPU

        // add new ligands to the cuda world

        let err = self.cuda_world.as_mut().unwrap().add_ligands(&self.new_ligands);

        self.new_ligands.clear();

        // error handling for adding ligands
        if let Err(_) = err {
            // increase capacity 
            self.cuda_world.as_mut().unwrap().increase_cap(crate::cuda::IncreaseType::Ligand);
            
        }

        // get the received ligands from the entities
        let (received_ligands, overflow) = self.cuda_world.as_mut().unwrap().update(&self.entities, self.space.max_size.ceil() as u32);

        if overflow > 0 {
            use crate::{cuda, edit_settings};

            eprintln!("Warning: Grid overflow occurred, increasing grid size and slots per cell; Overflow count: {}", overflow);
            let new_size = self.settings.cuda_slots_per_cell() * 2;

            // edit the settings to increase the grid size
            edit_settings!(self, cuda_slots_per_cell = new_size);

            // recreate the grid with the new size
            self.cuda_world.as_mut().unwrap().increase_cap(cuda::IncreaseType::Grid);
        }


        let len = received_ligands.count as usize;


        // check if the pointers are null
        if !received_ligands.receptor_ids.is_null() & !received_ligands.specs.is_null() {
            // slice around the *mut pointers
            let specs: &[u32];
            // receptor ids are stored as (entity_id * receptor_capacity + receptor_index)
            let receptors: &[u32];
            let entity_ids: &[u32];

            unsafe {
                specs = std::slice::from_raw_parts(received_ligands.specs, len);
                receptors = std::slice::from_raw_parts(received_ligands.receptor_ids, len * 2);
                entity_ids = std::slice::from_raw_parts(received_ligands.entity_ids, len);
            }

            // add the ligands to entities and edit concentrations
            for i in 0..len {
                let entity_index = entity_ids[i] as usize;
                let receptor_index = (receptors[i] % self.settings.receptors_per_entity() as u32) as usize;
                let spec = specs[i] as u16;
        

                if receptor_index == 0xFFFFFFFF {
                    // toxin
                    self.entities[entity_index].receive_toxins(spec, &self.settings);
                }

                // can go through the shortcut because the bond was already checked on the GPU
                self.entities[entity_index].receive_ligand_cuda_shortcut(spec, receptor_index, &self.settings);
                
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
            libc::free(received_ligands.specs as *mut libc::c_void);
            libc::free(received_ligands.entity_ids as *mut libc::c_void);
        }

        // output update on CPU 
        let dt = 1.0 / self.settings.fps() as f32;
        for entity in 0..self.entities.len() {
            self.entities[entity].update_output(&self.settings) 
        }


        // emit new ligands from entities
        for entity in &mut self.entities {
            let new_ligands = entity.emit_ligands(&self.settings);
            self.new_ligands.extend(new_ligands);
        }

        // emit new ligands from sources
        for source in &self.ligand_sources {
            let new_ligands = source.emit_ligands(dt, &self.settings);
            self.new_ligands.extend(new_ligands);

        }

        let entities_clone = self.entities.clone();

        // emit new entities
        for entity in 0..self.entities.len() {

            let new_entity = self.entities[entity].reproduce(self.counter, &mut self.space, &entities_clone, &self.settings);
            
            if let Some(mut e) = new_entity {
                e.cuda_receptor_index = Some(self.cuda_world.as_mut().unwrap().receptor_index());

                // add the receptors to the cuda world
                self.cuda_world.as_mut().unwrap().add_entity_receptors(&e);


                // add the entity to the world
                self.entities.push(e);
                self.counter += 1;


            }        
        }

        // remove dead entities
        self.entities.retain(|entity| entity.energy > 0.0);
        self.entities.retain(|entity| entity.age as f32 * dt < self.settings.max_age() as f32 || self.settings.max_age() == 0);

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


    pub fn delete_all_ligands(&mut self) {
        self.ligands.clear();

        #[cfg(feature = "cuda")]
        {
            if self.cuda_world.is_some() {
                self.cuda_world.as_mut().unwrap().delete_ligands();
            }
        }
    }
}

use std::io::{self, Write};
use std::fs::OpenOptions;
use std::io::Seek;


// save impl Block
impl World{

    
    // to be accessed by user
    // update the path where the world is saved
    pub fn save(&mut self, path: Option<&str>, save_genome: bool) -> io::Result<()>

    {
        let ex_path: std::path::PathBuf;
        if path.is_none() {
            ex_path = std::path::PathBuf::from(self.settings.path());
        } else {
            ex_path = std::path::PathBuf::from(path.unwrap());
        }

        // check if the path exists
        if ex_path.exists() {
            eprint!("Warning: File {} already exists. Overwrite? (y/n)", ex_path.display());
            let mut input = String::new();   
            std::io::stdin().read_line(&mut input).expect("Failed to read line");
            if input.trim().to_lowercase() != "y" {      
                return Err(io::Error::new(io::ErrorKind::Other, "File already exists"));
            }
            std::fs::remove_file(&ex_path)?;
        }


        self.path = Some(ex_path);
        self.save_genome = save_genome;
        self.save_header()?;
        Ok(())
    }

    pub fn is_saving(&self) -> bool {
        self.path.is_some()
    }



    pub fn stop_save(&mut self) -> Result<(), String> {
        self.path = None;
        Ok(())
    }

    fn save_buffer(&mut self) -> io::Result<()> {
        // Save the current buffer, containing serialized states of the world
        if self.path.is_none() {
            return Err(io::Error::new(io::ErrorKind::Other, "Save path is not set"));
        }
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

        self.pause_save()?;

        println!("State saved successfully");

        Ok(())
    }

    fn pause_save(&mut self) -> io::Result<()> {
        if self.path.is_none() {
            return Err(io::Error::new(io::ErrorKind::Other, "Save path is not set"));
        }

        // save the pause state -------------------------
        if let Ok(buffer) = self.pause_serialize(self.save_genome) {
            // append the pause state to the file
            let mut file = OpenOptions::new()
                .write(true)
                .append(true)
                .open(self.path.as_ref().unwrap())?;

            file.write_all(&buffer)?;
            
            self.byte_counter += buffer.len();

            // update save jumper in the header
            let mut file = OpenOptions::new()
                .write(true)
                .open(self.path.as_ref().unwrap())?;

            file.seek(std::io::SeekFrom::Start(1))?;
            file.write_all(&(self.byte_counter as u32).to_le_bytes())?;
        } else {
            eprintln!("Failed to serialize pause state");
        }

        Ok(())
    }


    fn save_table(&mut self) -> io::Result<()> {
        // allocates capacity for the jumper table to the file

        // copy the bytes written to this number for the jumper locations
        let mut bytes_written = self.byte_counter + (self.settings.store_capacity()+ 1) * 4; // add space for all the jumpers and the jumper for the next jumper

        let mut jumper_table = Vec::with_capacity((self.settings.store_capacity() + 1)* 4); // store capacity + save jumper + next jumper

        // fill the jumper_table with the addresses for the jumper
        for i in 0..self.settings.store_capacity(){
            let state_size = self.buffer[i].len();
            jumper_table.extend((bytes_written as u32).to_le_bytes());
            bytes_written += state_size;
        }

        // jumper to the next jumper table
        jumper_table.extend((bytes_written as u32).to_le_bytes());

        self.byte_counter = bytes_written;


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

        self.byte_counter += serialize::HEADER_SIZE as usize;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn copy_ligands(&mut self){
        use crate::cuda;

        self.ligands.clear(); 

        if self.cuda_world.is_none() {
            return ;
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

        unsafe { libc::free(ligands_h as *mut libc::c_void); }
        

    }

} 

#[cfg(feature = "debug")]
impl World {
    
    // add n ligands at random positions
    // only for testing purposes
    pub fn add_ligands(&mut self, n: usize) {
        if self.init == false {
            eprintln!("World not initialized, cannot add ligands");
            return;
        }
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
            let ligand = objects::Ligand::new(0, 0u16, position, norm_pos, &self.settings);
            self.new_ligands.push(ligand);
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
