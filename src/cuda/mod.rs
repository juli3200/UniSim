#![cfg(feature = "cuda")]
use crate::Settings;
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
            ligand_n: ligands.len() as u32,
            ligand_cap,
            ligands_pos: ligands_pos_d,
            ligands_vel: ligands_vel_d,
            ligands_content: ligands_content_d, // not yet implemented
        }
    }

    pub(crate) fn add_entities_to_grid(&self, entities: &Vec<Entity>) -> Result<(), String> {
        // checks if there is enough capacity
        if self.entity_n + entities.len() as u32 > self.entity_cap { 
            return Err("Entity capacity exceeded".into());
        }

        use cuda_bindings::memory_gpu as cu_mem;

        // _______________________ copy new entities to device memory _______________________

        Ok(())
    }
}