#![cfg(feature = "cuda")]
use crate::{objects, Settings};
use crate::objects::{Entity, Ligand};

pub(crate) mod cuda_bindings;


const EXTRA_SPACE_ENTITY: f32 = 1.2; // allocate 20% more space than needed for entities
const EXTRA_SPACE_LIGAND: f32 = 2.0; // allocate 100% more space than needed for ligands
const MIN_SPACE_LIGAND: usize = 1_000_000; // minimum space for ligands
const BUFFER_SIZE: usize = 10 * 1024 * 1024; // 10 MB buffer for saving data from GPU (remove const later)

/// CUDA-related structures and functions
/// e.g as CUDA arrays, kernels links, etc.

// if gpu is active positions, velocities, and sizes of the objects are each stored in a single array on gpu memory
// this is to minimize the number of memory transfers between cpu and gpu
#[derive(Debug, Clone)]
pub(crate) struct CUDAWorld{

    pub(crate) settings: Settings,

    // The following are pointers to CUDA device memory

    // 3D Array: grid[x][y][i] = index of the ligand at that position, or 0 if empty
    // the third dimension is a list of indices, to allow multiple ligands in the same cell
    // the size of the third dimension is fixed, settings.cuda_slots_per_cell
    grid: *mut u32,

    // Buffer to save data from GPU to CPU
    save_buffer: *mut u8,

    // number and capacity of entities
    entity_n: u32,
    entity_cap: u32,

    // Entity data arrays
    entities_pos: *mut f32,
    entities_vel: *mut f32,
    entities_acc: *mut f32,
    entities_size: *mut f32,
    entities_id: *mut u32,
    entities_cell: *mut u32, // cell positions for each entity in grid

    // number and capacity of ligands
    ligand_n: u32,
    ligand_cap: u32,

    // Ligand data arrays
    ligands_pos: *mut f32,
    ligands_vel: *mut f32,
    ligands_content: *mut u32, // not yet implemented
}

impl CUDAWorld {
    pub(crate) fn new(settings: &Settings, entities: &Vec<Entity>, ligands: &Vec<Ligand>) -> Self {

        let entity_cap = (((entities.len()).max(settings.default_population()) as f32) * EXTRA_SPACE_ENTITY) as u32; 
        let ligand_cap = (((ligands.len()).max(MIN_SPACE_LIGAND) as f32) * EXTRA_SPACE_LIGAND) as u32; 

        // create host-side vectors to hold data before copying to device
        let mut entities_pos_h = Vec::with_capacity((entity_cap * 2) as usize);
        let mut entities_vel_h = Vec::with_capacity((entity_cap * 2) as usize);
        let mut entities_acc_h = Vec::with_capacity((entity_cap * 2) as usize);

        let mut entities_size_h = Vec::with_capacity((entity_cap) as usize);
        let mut entities_id_h = Vec::with_capacity((entity_cap) as usize);
        let mut entities_cell_h = Vec::with_capacity((entity_cap) as usize);

        for entity in entities {
            entities_pos_h.push(entity.position[0]);
            entities_pos_h.push(entity.position[1]);

            entities_vel_h.push(entity.velocity[0]);
            entities_vel_h.push(entity.velocity[1]);

            entities_acc_h.push(entity.acceleration[0]);
            entities_acc_h.push(entity.acceleration[1]);

            entities_size_h.push(entity.size);
            entities_id_h.push(entity.id as u32);
            entities_cell_h.push(0u32); // initialize cell positions to zero
        }


        let mut ligands_pos_h = Vec::with_capacity((ligand_cap * 2) as usize);
        let mut ligands_vel_h = Vec::with_capacity((ligand_cap * 2) as usize);
        let mut ligands_content_h = Vec::with_capacity((ligand_cap) as usize); // not yet implemented
        for ligand in ligands {
            ligands_pos_h.push(ligand.position[0]);
            ligands_pos_h.push(ligand.position[1]);

            ligands_vel_h.push(ligand.velocity[0]);
            ligands_vel_h.push(ligand.velocity[1]);

            // --------------------------------------------------------------------------- not yet implemented ---------------------------------------------------------------------------
            ligands_content_h.push(0u32);
        }

        // import cuda bindings with shorter names
        use cuda_bindings::memory_gpu as cu_mem;
        use cuda_bindings::grid_gpu as cu_grid;

        // define device pointers so they can be used in unsafe block and still be accessible later
        let grid_d: *mut u32; 
        let save_buffer_d: *mut u8;

        let entities_pos_d: *mut f32; 
        let entities_vel_d: *mut f32; 
        let entities_acc_d: *mut f32; 
        let entities_size_d: *mut f32; 
        let entities_id_d: *mut u32; 
        let entities_cell_d: *mut u32;

        let ligands_pos_d: *mut f32; 
        let ligands_vel_d: *mut f32; 
        let ligands_content_d: *mut u32; // not yet implemented

        // allocate device memory and copy data
        unsafe{
            // ----------------- grid -----------------
            let grid_size = settings.dimensions().0 * settings.dimensions().1 * settings.cuda_slots_per_cell() as u32;

            // allocate grid and set to zero
            grid_d = cu_mem::alloc_u(grid_size);
            cu_grid::clear_grid(grid_d, grid_size);


            // ----------------- save buffer -----------------
            // allocate save buffer and set to zero
            save_buffer_d = cu_mem::alloc_c(BUFFER_SIZE as u32);
            cu_mem::copy_HtoD_c(save_buffer_d, vec![0u8; BUFFER_SIZE].as_mut_ptr(), BUFFER_SIZE as u32);


            // ----------------- entities -----------------
            let size_entity = entities.len() as u32 * 2; 

            // positions, allocate double the space for x and y of capacity
            entities_pos_d = cu_mem::alloc_f(entity_cap * 2);
            cu_mem::copy_HtoD_f(entities_pos_d, entities_pos_h.as_mut_ptr(), size_entity); // data size is number of entities * 2, so there is space for more

            // velocities
            entities_vel_d = cu_mem::alloc_f(entity_cap * 2);
            cu_mem::copy_HtoD_f(entities_vel_d, entities_vel_h.as_mut_ptr(), size_entity);

            // accelerations
            entities_acc_d = cu_mem::alloc_f(entity_cap * 2);
            cu_mem::copy_HtoD_f(entities_acc_d, entities_acc_h.as_mut_ptr(), size_entity);

            // sizes
            entities_size_d = cu_mem::alloc_f(entity_cap);
            cu_mem::copy_HtoD_f(entities_size_d, entities_size_h.as_mut_ptr(), size_entity / 2);

            // ids
            entities_id_d = cu_mem::alloc_u(entity_cap);
            cu_mem::copy_HtoD_u(entities_id_d, entities_id_h.as_mut_ptr(), size_entity / 2);

            // cell positions
            entities_cell_d = cu_mem::alloc_u(entity_cap);
            cu_mem::copy_HtoD_u(entities_cell_d, entities_cell_h.as_mut_ptr(), size_entity / 2);


            // ----------------- ligands -----------------
            // positions, allocate double the space for x and y of capacity
            let size_ligand = ligands.len() as u32 * 2;

            // positions
            ligands_pos_d = cu_mem::alloc_f(ligand_cap * 2);
            cu_mem::copy_HtoD_f(ligands_pos_d, ligands_pos_h.as_mut_ptr(), size_ligand);

            // velocities
            ligands_vel_d = cu_mem::alloc_f(ligand_cap * 2);
            cu_mem::copy_HtoD_f(ligands_vel_d, ligands_vel_h.as_mut_ptr(), size_ligand);

            // contents, not yet implemented
            ligands_content_d = cu_mem::alloc_u(ligand_cap);
            cu_mem::copy_HtoD_u(ligands_content_d, ligands_content_h.as_mut_ptr(), size_ligand / 2);

        }


        Self {
            settings: settings.clone(),
            grid: grid_d,
            save_buffer: save_buffer_d,
            entity_n: entities.len() as u32,
            entity_cap,
            entities_pos: entities_pos_d,
            entities_vel: entities_vel_d,
            entities_acc: entities_acc_d,
            entities_size: entities_size_d,
            entities_id: entities_id_d,
            entities_cell: entities_cell_d,
            ligand_n: ligands.len() as u32,
            ligand_cap,
            ligands_pos: ligands_pos_d,
            ligands_vel: ligands_vel_d,
            ligands_content: ligands_content_d, // not yet implemented
        }
    }

    pub(crate) fn add_entities(&self, entities: &Vec<Entity>) -> Result<(), String> {
        // checks if there is enough capacity
        if self.entity_n + entities.len() as u32 > self.entity_cap { 
            return Err("Entity capacity exceeded".into());
        }

        use cuda_bindings::memory_gpu as cu_mem;

        // create host-side vectors to hold data before copying to device
        let mut entities_pos_h = Vec::with_capacity(entities.len() * 2);
        let mut entities_vel_h = Vec::with_capacity(entities.len() * 2);
        let mut entities_acc_h = Vec::with_capacity(entities.len() * 2);
        let mut entities_size_h = Vec::with_capacity(entities.len());
        let mut entities_id_h = Vec::with_capacity(entities.len());

        for entity in entities {
            entities_pos_h.push(entity.position[0]);
            entities_pos_h.push(entity.position[1]);

            entities_vel_h.push(entity.velocity[0]);
            entities_vel_h.push(entity.velocity[1]);

            entities_acc_h.push(entity.acceleration[0]);
            entities_acc_h.push(entity.acceleration[1]);

            entities_size_h.push(entity.size);
            entities_id_h.push(entity.id as u32);
        }

        unsafe{
            let start_index = self.entity_n as usize;
            let size_entity = entities.len() as u32 * 2; 

            // positions
            cu_mem::copy_HtoD_f(self.entities_pos.add(start_index * 2), entities_pos_h.as_mut_ptr(), size_entity);

            // velocities
            cu_mem::copy_HtoD_f(self.entities_vel.add(start_index * 2), entities_vel_h.as_mut_ptr(), size_entity);

            // accelerations
            cu_mem::copy_HtoD_f(self.entities_acc.add(start_index * 2), entities_acc_h.as_mut_ptr(), size_entity);

            // sizes
            cu_mem::copy_HtoD_f(self.entities_size.add(start_index), entities_size_h.as_mut_ptr(), size_entity / 2);

            // ids
            cu_mem::copy_HtoD_u(self.entities_id.add(start_index), entities_id_h.as_mut_ptr(), size_entity / 2);
        }


        Ok(())
    }

    pub(crate) fn add_ligands(&self, ligands_pos: &mut Vec<f32>, ligands_vel: &mut Vec<f32>, ligands_content: &mut Vec<u32>) -> Result<(), i32> {
        // checks if input vectors are of correct size
        if ligands_pos.len() != ligands_vel.len() || ligands_pos.len() / 2 != ligands_content.len() {
            return Err(-1);
        }
        
        // checks if there is enough capacity
        if self.ligand_n + ligands_content.len() as u32 > self.ligand_cap { 
            return Err(-2);
        }

        use cuda_bindings::memory_gpu as cu_mem;


        unsafe{
            let start_index = self.ligand_n as usize;
            let size_ligand = ligands_pos.len() as u32;

            // positions
            cu_mem::copy_HtoD_f(self.ligands_pos.add(start_index * 2), ligands_pos.as_mut_ptr(), size_ligand);

            // velocities
            cu_mem::copy_HtoD_f(self.ligands_vel.add(start_index * 2), ligands_vel.as_mut_ptr(), size_ligand);

            // contents, not yet implemented
            cu_mem::copy_HtoD_u(self.ligands_content.add(start_index), ligands_content.as_mut_ptr(), size_ligand / 2);
        }
        Ok(())

    }

    pub(crate) fn add_to_grid(&self, start_index: u32, end_index: u32) -> i32{
        use cuda_bindings::grid_gpu as cu_grid;
        use cuda_bindings::memory_gpu as cu_mem;

        let size = end_index - start_index;

        let overflow;

        unsafe{
            // create dimension array
            let dim = cu_mem::alloc_u(3);
            let mut h_dim = vec![self.settings.dimensions().0, self.settings.dimensions().1, self.settings.cuda_slots_per_cell() as u32];
            cu_mem::copy_HtoD_u(dim, h_dim.as_mut_ptr(), 3);

            overflow = cu_grid::fill_grid(size, dim, self.grid, self.entities_pos, self.entities_cell);
            
            // free dimension array
            cu_mem::free_u(dim);
        }

        return overflow;
        
    }

    pub(crate) fn increase_cap(&mut self, obj_type: objects::ObjectType) {

        use cuda_bindings::memory_gpu as cu_mem;

        unsafe {
        match obj_type {
            objects::ObjectType::Entity(_) => {
                let new_cap = (self.entity_cap as f32 * EXTRA_SPACE_ENTITY) as u32;
                self.entity_cap = new_cap;

                println!("Increasing entity capacity to {}", new_cap);
                
                // reallocate device memory
                
                // positions
                let new_entities_pos = cu_mem::alloc_f(new_cap * 2);
                cu_mem::copy_DtoD_f(new_entities_pos, self.entities_pos, self.entity_n * 2);
                cu_mem::free_f(self.entities_pos);
                self.entities_pos = new_entities_pos;

                // velocities
                let new_entities_vel = cu_mem::alloc_f(new_cap * 2);
                cu_mem::copy_DtoD_f(new_entities_vel, self.entities_vel, self.entity_n * 2);
                cu_mem::free_f(self.entities_vel);
                self.entities_vel = new_entities_vel;

                // accelerations
                let new_entities_acc = cu_mem::alloc_f(new_cap * 2);
                cu_mem::copy_DtoD_f(new_entities_acc, self.entities_acc, self.entity_n * 2);
                cu_mem::free_f(self.entities_acc);
                self.entities_acc = new_entities_acc;

                // sizes
                let new_entities_size = cu_mem::alloc_f(new_cap);
                cu_mem::copy_DtoD_f(new_entities_size, self.entities_size, self.entity_n);
                cu_mem::free_f(self.entities_size);
                self.entities_size = new_entities_size;

                // ids
                let new_entities_id = cu_mem::alloc_u(new_cap);
                cu_mem::copy_DtoD_u(new_entities_id, self.entities_id, self.entity_n);
                cu_mem::free_u(self.entities_id);
                self.entities_id = new_entities_id;

                // cell positions
                let new_entities_cell = cu_mem::alloc_u(new_cap);
                cu_mem::copy_DtoD_u(new_entities_cell, self.entities_cell, self.entity_n);
                cu_mem::free_u(self.entities_cell);
                self.entities_cell = new_entities_cell;



            },
            objects::ObjectType::Ligand(_) => {
                let new_cap = (self.ligand_cap as f32 * EXTRA_SPACE_LIGAND) as u32;
                self.ligand_cap = new_cap;

                println!("Increasing ligand capacity to {}", new_cap);
                
                // reallocate device memory

                // positions
                let new_ligands_pos = cu_mem::alloc_f(new_cap * 2);
                cu_mem::copy_DtoD_f(new_ligands_pos, self.ligands_pos, self.ligand_n * 2);
                cu_mem::free_f(self.ligands_pos);
                self.ligands_pos = new_ligands_pos;
                
                // velocities
                let new_ligands_vel = cu_mem::alloc_f(new_cap * 2);
                cu_mem::copy_DtoD_f(new_ligands_vel, self.ligands_vel, self.ligand_n * 2);
                cu_mem::free_f(self.ligands_vel);
                self.ligands_vel = new_ligands_vel;

                // contents, not yet implemented
                let new_ligands_content = cu_mem::alloc_u(new_cap);
                cu_mem::copy_DtoD_u(new_ligands_content, self.ligands_content, self.ligand_n);
                cu_mem::free_u(self.ligands_content);
                self.ligands_content = new_ligands_content;
            },
        }
        }
    }
}