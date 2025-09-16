use super::cuda_bindings;
use super::*;

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
        let mut ligands_message_h = Vec::with_capacity((ligand_cap) as usize); // not yet implemented
        for ligand in ligands {
            ligands_pos_h.push(ligand.position[0]);
            ligands_pos_h.push(ligand.position[1]);

            ligands_vel_h.push(ligand.velocity[0]);
            ligands_vel_h.push(ligand.velocity[1]);

            // --------------------------------------------------------------------------- not yet implemented ---------------------------------------------------------------------------
            ligands_message_h.push(ligand.message);
        }

        // import cuda bindings with shorter names
        use cuda_bindings::memory_gpu as cu_mem;

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
        let ligands_message_d: *mut u32; // not yet implemented

        // allocate device memory and copy data
        unsafe{
            // ----------------- grid -----------------
            let grid_size = settings.dimensions().0 * settings.dimensions().1 * settings.cuda_slots_per_cell() as u32;

            // allocate grid and set to zero
            grid_d = cu_mem::alloc_u(grid_size);
            cu_mem::clear_u(grid_d, grid_size);


            // ----------------- save buffer -----------------
            // allocate save buffer and set to zero
            save_buffer_d = cu_mem::alloc_c(BUFFER_SIZE as u32);
            cu_mem::clear_c(save_buffer_d, BUFFER_SIZE as u32);
            cu_mem::copy_HtoD_c(save_buffer_d, vec![0u8; BUFFER_SIZE].as_mut_ptr(), BUFFER_SIZE as u32);


            // ----------------- entities -----------------
            let size_entity = entities.len() as u32 * 2; 

            // positions, allocate double the space for x and y of capacity
            entities_pos_d = cu_mem::alloc_f(entity_cap * 2);
            cu_mem::clear_f(entities_pos_d, entity_cap * 2);
            cu_mem::copy_HtoD_f(entities_pos_d, entities_pos_h.as_mut_ptr(), size_entity); // data size is number of entities * 2, so there is space for more

            // velocities
            entities_vel_d = cu_mem::alloc_f(entity_cap * 2);
            cu_mem::clear_f(entities_vel_d, entity_cap * 2);
            cu_mem::copy_HtoD_f(entities_vel_d, entities_vel_h.as_mut_ptr(), size_entity);

            // accelerations
            entities_acc_d = cu_mem::alloc_f(entity_cap * 2);
            cu_mem::clear_f(entities_acc_d, entity_cap * 2);
            cu_mem::copy_HtoD_f(entities_acc_d, entities_acc_h.as_mut_ptr(), size_entity);

            // sizes
            entities_size_d = cu_mem::alloc_f(entity_cap);
            cu_mem::clear_f(entities_size_d, entity_cap);
            cu_mem::copy_HtoD_f(entities_size_d, entities_size_h.as_mut_ptr(), size_entity / 2);

            // ids
            entities_id_d = cu_mem::alloc_u(entity_cap);
            cu_mem::clear_u(entities_id_d, entity_cap);
            cu_mem::copy_HtoD_u(entities_id_d, entities_id_h.as_mut_ptr(), size_entity / 2);

            // cell positions
            entities_cell_d = cu_mem::alloc_u(entity_cap);
            cu_mem::clear_u(entities_cell_d, entity_cap);
            cu_mem::copy_HtoD_u(entities_cell_d, entities_cell_h.as_mut_ptr(), size_entity / 2);


            // ----------------- ligands -----------------
            // positions, allocate double the space for x and y of capacity
            let size_ligand = ligands.len() as u32 * 2;

            // positions
            ligands_pos_d = cu_mem::alloc_f(ligand_cap * 2);
            cu_mem::clear_f(ligands_pos_d, ligand_cap * 2);
            cu_mem::copy_HtoD_f(ligands_pos_d, ligands_pos_h.as_mut_ptr(), size_ligand);

            // velocities
            ligands_vel_d = cu_mem::alloc_f(ligand_cap * 2);
            cu_mem::clear_f(ligands_vel_d, ligand_cap * 2);
            cu_mem::copy_HtoD_f(ligands_vel_d, ligands_vel_h.as_mut_ptr(), size_ligand);

            // messages
            ligands_message_d = cu_mem::alloc_u(ligand_cap);
            cu_mem::clear_u(ligands_message_d, ligand_cap);
            cu_mem::copy_HtoD_u(ligands_message_d, ligands_message_h.as_mut_ptr(), size_ligand / 2);

        }

        // create EntityArrays and LigandArrays structs
        let entities = EntityArrays {
            pos: entities_pos_d,
            vel: entities_vel_d,
            acc: entities_acc_d,
            size: entities_size_d,
            id: entities_id_d,
            cell: entities_cell_d,
            num_entities: entities.len(),
        };

        let ligands = LigandArrays {
            pos: ligands_pos_d,
            vel: ligands_vel_d,
            message: ligands_message_d,
            num_ligands: ligands.len(),
        };



        Self {
            settings: settings.clone(),
            grid: grid_d,
            save_buffer: save_buffer_d,
            entity_cap,
            entities,
            ligand_cap,
            ligands,
        }
    }

    pub(crate) fn add_entities(&mut self, entities: &Vec<Entity>) -> Result<(), String> {
        // checks if there is enough capacity
        if self.entities.num_entities + entities.len()  > self.entity_cap as usize {
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
            let start_index = self.entities.num_entities;
            let size_entity = entities.len() as u32 * 2;

            // positions
            cu_mem::copy_HtoD_f(self.entities.pos.add(start_index * 2), entities_pos_h.as_mut_ptr(), size_entity);

            // velocities
            cu_mem::copy_HtoD_f(self.entities.vel.add(start_index * 2), entities_vel_h.as_mut_ptr(), size_entity);

            // accelerations
            cu_mem::copy_HtoD_f(self.entities.acc.add(start_index * 2), entities_acc_h.as_mut_ptr(), size_entity);

            // sizes
            cu_mem::copy_HtoD_f(self.entities.size.add(start_index), entities_size_h.as_mut_ptr(), size_entity / 2);

            // ids
            cu_mem::copy_HtoD_u(self.entities.id.add(start_index), entities_id_h.as_mut_ptr(), size_entity / 2);
        }

        self.entities.num_entities += entities.len();
        Ok(())
    }

    pub(crate) fn add_ligands(&mut self, ligands_pos: &mut Vec<f32>, ligands_vel: &mut Vec<f32>, ligands_content: &mut Vec<u32>) -> Result<(), i32> {
        // checks if input vectors are of correct size
        if ligands_pos.len() != ligands_vel.len() || ligands_pos.len() / 2 != ligands_content.len() {
            return Err(-1);
        }
        
        // checks if there is enough capacity
        if self.ligands.num_ligands + ligands_content.len() > self.ligand_cap as usize {
            return Err(-2);
        }

        use cuda_bindings::memory_gpu as cu_mem;


        unsafe{
            let start_index = self.ligands.num_ligands;
            let size_ligand = ligands_pos.len() as u32;

            // positions
            cu_mem::copy_HtoD_f(self.ligands.pos.add(start_index * 2), ligands_pos.as_mut_ptr(), size_ligand);

            // velocities
            cu_mem::copy_HtoD_f(self.ligands.vel.add(start_index * 2), ligands_vel.as_mut_ptr(), size_ligand);

            // messages
            cu_mem::copy_HtoD_u(self.ligands.message.add(start_index), ligands_content.as_mut_ptr(), size_ligand / 2);
        }

        self.ligands.num_ligands += ligands_content.len();

        Ok(())

    }

    pub(crate) fn add_to_grid(&self, start_index: u32, end_index: u32) -> i32{
        use cuda_bindings::grid_gpu as cu_grid;

        let size = end_index - start_index;

        let overflow;

        unsafe{
            let dim = Dim{
                x: self.settings.dimensions().0,
                y: self.settings.dimensions().1,
                depth: self.settings.cuda_slots_per_cell() as u32,
            };

            overflow = cu_grid::fill_grid(size, dim, self.grid, self.entities.pos, self.entities.cell);
            
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
                cu_mem::clear_f(new_entities_pos, new_cap * 2);
                cu_mem::copy_DtoD_f(new_entities_pos, self.entities.pos, self.entities.num_entities as u32  * 2);
                cu_mem::free_f(self.entities.pos);
                self.entities.pos = new_entities_pos;

                // velocities
                let new_entities_vel = cu_mem::alloc_f(new_cap * 2);
                cu_mem::clear_f(new_entities_vel, new_cap * 2);
                cu_mem::copy_DtoD_f(new_entities_vel, self.entities.vel, self.entities.num_entities as u32 * 2);
                cu_mem::free_f(self.entities.vel);
                self.entities.vel = new_entities_vel;

                // accelerations
                let new_entities_acc = cu_mem::alloc_f(new_cap * 2);
                cu_mem::clear_f(new_entities_acc, new_cap * 2);
                cu_mem::copy_DtoD_f(new_entities_acc, self.entities.acc, self.entities.num_entities as u32 * 2);
                cu_mem::free_f(self.entities.acc);
                self.entities.acc = new_entities_acc;

                // sizes
                let new_entities_size = cu_mem::alloc_f(new_cap);
                cu_mem::clear_f(new_entities_size, new_cap);
                cu_mem::copy_DtoD_f(new_entities_size, self.entities.size, self.entities.num_entities as u32);
                cu_mem::free_f(self.entities.size);
                self.entities.size = new_entities_size;

                // ids
                let new_entities_id = cu_mem::alloc_u(new_cap);
                cu_mem::clear_u(new_entities_id, new_cap);
                cu_mem::copy_DtoD_u(new_entities_id, self.entities.id, self.entities.num_entities as u32);
                cu_mem::free_u(self.entities.id);
                self.entities.id = new_entities_id;

                // cell positions
                let new_entities_cell = cu_mem::alloc_u(new_cap);
                cu_mem::clear_u(new_entities_cell, new_cap);
                cu_mem::copy_DtoD_u(new_entities_cell, self.entities.cell, self.entities.num_entities as u32);
                cu_mem::free_u(self.entities.cell);
                self.entities.cell = new_entities_cell;



            },
            objects::ObjectType::Ligand(_) => {
                let new_cap = (self.ligand_cap as f32 * EXTRA_SPACE_LIGAND) as u32;
                self.ligand_cap = new_cap;

                println!("Increasing ligand capacity to {}", new_cap);
                
                // reallocate device memory

                // positions
                let new_ligands_pos = cu_mem::alloc_f(new_cap * 2);
                cu_mem::copy_DtoD_f(new_ligands_pos, self.ligands.pos, self.ligands.num_ligands as u32 * 2);
                cu_mem::free_f(self.ligands.pos);
                self.ligands.pos = new_ligands_pos;

                // velocities
                let new_ligands_vel = cu_mem::alloc_f(new_cap * 2);
                cu_mem::copy_DtoD_f(new_ligands_vel, self.ligands.vel, self.ligands.num_ligands as u32 * 2);
                cu_mem::free_f(self.ligands.vel);
                self.ligands.vel = new_ligands_vel;

                // contents, not yet implemented
                let new_ligands_message = cu_mem::alloc_u(new_cap);
                cu_mem::copy_DtoD_u(new_ligands_message, self.ligands.message, self.ligands.num_ligands as u32);
                cu_mem::free_u(self.ligands.message);
                self.ligands.message = new_ligands_message;
            },
        }
        }
    }

    pub(crate) fn update(&mut self, search_radius: u32) -> CollisionArraysHost{

        // first, update ligand positions based on their velocities
        use cuda_bindings::grid_gpu as cu_grid;

        unsafe{
            let delta_time = 1.0 / self.settings.fps();
            cu_grid::update_positions(self.ligands.clone(), delta_time);
        }

        // then, handle collisions
        let collisions;
        unsafe {
            let dim = Dim {
                x: self.settings.dimensions().0,
                y: self.settings.dimensions().1,
                depth: self.settings.cuda_slots_per_cell() as u32,
            };
            collisions = cu_grid::ligand_collision(search_radius, dim, self.grid, self.entities.clone(), self.ligands.clone());
        }

        return collisions;
    }
}