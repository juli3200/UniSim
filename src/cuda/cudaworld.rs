use super::cuda_bindings;
use super::*;
use ndarray::Array1;


use crate::objects::receptor::sequence_receptor;


impl EntityCuda {
    pub fn from_entity(entity: &Entity) -> Self {
        Self {
            posx: entity.position[0],
            posy: entity.position[1],
            velx: entity.velocity[0],
            vely: entity.velocity[1],
            size: entity.size,
            receptor_id: entity.cuda_receptor_index.unwrap_or(0),
        }
    }
}


impl LigandCuda {
    fn from_ligand(ligand: &Ligand, settings: &Settings) -> Self {
        Self {
            emitted_id: ligand.emitted_id as u32,
            posx: ligand.position[0],
            posy: ligand.position[1],
            velx: ligand.velocity[0] * settings.ligand_velocity(),
            vely: ligand.velocity[1] * settings.ligand_velocity(),
            spec: ligand.spec as u32,
            energy: ligand.energy,
        }
    }
}

impl TryInto<Ligand> for &LigandCuda {
    type Error = &'static str;

    fn try_into(self) -> Result<Ligand, Self::Error> {
        if self.emitted_id == 0xFFFFFFFF {
            return Err("Ligand was deleted");
            
        }
        Ok(Ligand {
            emitted_id: self.emitted_id as usize,
            position: Array1::from_vec(vec![self.posx, self.posy]),
            velocity: Array1::from_vec(vec![self.velx, self.vely]),
            spec: self.spec as u16,
            energy: self.energy,
        })
    }
}

impl CUDAWorld {
    pub(crate) fn new(settings: &Settings, entities: &Vec<Entity>, ligands: &Vec<Ligand>) -> Self {

        let entity_cap = (((entities.len()).max(settings.default_population()) as f32) * EXTRA_SPACE_ENTITY) as u32; 
        let ligand_cap = (((ligands.len()).max(MIN_SPACE_LIGAND) as f32) * EXTRA_SPACE_LIGAND) as u32; 


        // import cuda bindings with shorter name
        use cuda_bindings::memory_gpu as cu_mem;
        use cuda_bindings::grid_gpu as cu_grid;

        // define device pointers so they can be used in unsafe block and still be accessible later
        let grid_d: *mut u32; 
        let entities_d: *mut EntityCuda; // device pointer to entity array
        let receptors_d: *mut u32; // device pointer to receptor array
        let ligands_d: *mut LigandCuda; // device pointer to ligand array

        // allocate device memory
        unsafe{
            // ----------------- grid -----------------
            let grid_size = settings.dimensions().0 * settings.dimensions().1 * settings.cuda_slots_per_cell() as u32;
            // allocate grid and set to zero
            grid_d = cu_mem::alloc_u32(grid_size);
            cu_grid::clear_grid(grid_d, grid_size);


            // ----------------- entities -----------------
            entities_d = cu_mem::alloc_entity(entity_cap);

            // ----------------- receptors -----------------
            receptors_d = cu_mem::alloc_u32(entity_cap * settings.receptor_capacity() as u32);


            // ----------------- ligands -----------------
            // positions, allocate double the space for x and y of capacity
            ligands_d = cu_mem::alloc_ligand(ligand_cap);

        }


        let mut cudaw = Self {
            settings: settings.clone(),
            grid: grid_d,
            entity_cap,
            entity_count: entities.len() as u32,
            entities: entities_d,
            receptors: receptors_d,
            receptor_index: entities.len() as u32,
            ligand_count: ligands.len() as u32,
            ligand_cap,
            ligands: ligands_d,
        };

        // fill data arrays
        cudaw.fill_receptors(entities);


        return cudaw;


    }

    pub(crate) fn free(&mut self) {
        use cuda_bindings::memory_gpu as cu_mem;

        unsafe{
            // free grid
            cu_mem::free_u32(self.grid);
            cu_mem::free_entity(self.entities);
            cu_mem::free_u32(self.receptors);
            cu_mem::free_ligand(self.ligands);
        }
    }

    pub(crate) fn receptor_index(&mut self) -> u32 {
        // returns current receptor index and increments it
        self.receptor_index += 1;
        self.receptor_index - 1
    }

    fn fill_receptors(&mut self, entities: &Vec<Entity>) {
        use cuda_bindings::memory_gpu as cu_mem;

        let size_receptors = (entities.len() * self.settings.receptor_capacity()) as u32;

        // create host-side vector to hold data before copying to device
        let mut receptors_h = Vec::with_capacity(entities.len() * self.settings.receptor_capacity());

        for entity in entities {
            for receptor in &entity.receptors {
                let (_, _ , spec) = sequence_receptor(*receptor);
                receptors_h.push(spec as u32);
            }
        }

        assert_eq!(receptors_h.len(), size_receptors as usize);

        // copy to device
        unsafe{
            cu_mem::copy_HtoD_u32(self.receptors, receptors_h.as_ptr(), size_receptors);
        }
    }

    pub(crate) fn update_entities(&mut self, entities: &Vec<Entity>) {
        // checks if there is enough capacity
        if entities.len()  > self.entity_cap as usize {
            self.increase_cap(IncreaseType::Entity);
        }

        use cuda_bindings::memory_gpu as cu_mem;

        // create host-side vectors to hold data before copying to device
        let mut entities_h: Vec<EntityCuda> = entities.iter().map(|e| EntityCuda::from_entity(e)).collect();
        
        unsafe {
            let size_entity = entities.len() as u32;
            cu_mem::copy_HtoD_entity(self.entities, entities_h.as_mut_ptr(), size_entity);
        }

    }

    pub(crate) fn add_ligands(&mut self, ligands: &Vec<Ligand>) -> Result<(), ()> {
        
        // checks if there is enough capacity
        if self.ligand_count as usize + ligands.len() > self.ligand_cap as usize {
            return Err(());
        }
        
        use cuda_bindings::memory_gpu as cu_mem;
        // make CudaLigand array
        let ligands_cuda: Vec<LigandCuda> = ligands.iter().map(|l| LigandCuda::from_ligand(l, &self.settings)).collect();

        unsafe{
            let start_index = self.ligand_count as usize;
            let size_ligand = ligands.len() as u32;

            cu_mem::copy_HtoD_ligand(self.ligands.add(start_index), ligands_cuda.as_ptr(), size_ligand);
        }

        self.ligand_count += ligands.len() as u32;

        Ok(())

    }

    pub(crate) fn add_to_grid(&self) -> i32{
        use cuda_bindings::grid_gpu as cu_grid;

        let overflow;

        unsafe{
            let dim = Dim{
                x: self.settings.dimensions().0,
                y: self.settings.dimensions().1,
                depth: self.settings.cuda_slots_per_cell() as u32,
            };

            overflow = cu_grid::fill_grid(self.entity_count, dim, self.grid, self.entities);
        }

        return overflow;
        
    }

    pub(crate) fn increase_cap(&mut self, obj_type: IncreaseType) {

        use cuda_bindings::memory_gpu as cu_mem;
        use cuda_bindings::grid_gpu as cu_grid;

        unsafe {
        match obj_type {
            IncreaseType::Entity => {
                let new_cap = (self.entity_cap as f32 * EXTRA_SPACE_ENTITY) as u32;
                self.entity_cap = new_cap;

                println!("Increasing entity capacity to {}", new_cap);
                
                // reallocate device memory
                let new_entities = cu_mem::alloc_entity(new_cap);
                cu_mem::copy_DtoD_entity(new_entities, self.entities, self.entity_count as u32);
                cu_mem::free_entity(self.entities);

                // reallocate receptors
                let new_receptors = cu_mem::alloc_u32(new_cap * self.settings.receptor_capacity() as u32);
                cu_mem::copy_DtoD_u32(new_receptors, self.receptors, self.entity_count as u32 * self.settings.receptor_capacity() as u32);
                cu_mem::free_u32(self.receptors);


                self.entities = new_entities;
                self.receptors = new_receptors;

            },
            IncreaseType::Ligand => {
                let new_cap = (self.ligand_cap as f32 * EXTRA_SPACE_LIGAND) as u32;
                self.ligand_cap = new_cap;

                println!("Increasing ligand capacity to {}", new_cap);
                
                // reallocate device memory
                let new_ligands = cu_mem::alloc_ligand(new_cap);
                cu_mem::copy_DtoD_ligand(new_ligands, self.ligands, self.ligand_count as u32);
                cu_mem::free_ligand(self.ligands);
                self.ligands = new_ligands;


            },
            IncreaseType::Grid => {
                
                let grid_size = self.settings.dimensions().0 * self.settings.dimensions().1 * self.settings.cuda_slots_per_cell() as u32;

                // free old grid
                cu_mem::free_u32(self.grid);

                // allocate new grid and set to zero
                self.grid = cu_mem::alloc_u32(grid_size);
                cu_grid::clear_grid(self.grid, grid_size);
            }
        }
        }
    }

    pub(crate) fn add_entity_receptors(&mut self, entity: &Entity) {
        use cuda_bindings::memory_gpu as cu_mem;

        if self.receptor_index >= self.entity_cap {
            self.increase_cap(IncreaseType::Entity);
        }

        let start_index = entity.cuda_receptor_index.unwrap() * self.settings.receptor_capacity() as u32;

        // create host-side vector to hold data before copying to device
        let mut receptors_h = Vec::with_capacity(self.settings.receptor_capacity());

        for receptor in &entity.receptors {
            let (_, _ , spec) = sequence_receptor(*receptor);
            receptors_h.push(spec as u32);
        }

        assert_eq!(receptors_h.len(), self.settings.receptor_capacity());

        // copy to device
        unsafe{
            cu_mem::copy_HtoD_u32(self.receptors.add(start_index as usize), receptors_h.as_ptr(), self.settings.receptor_capacity() as u32);
        }
    }

    pub(crate) fn update(&mut self, entities: &Vec<Entity>, search_radius: u32) -> (LigandWrapper, i32){

        // first, update ligand positions based on their velocities
        use cuda_bindings::grid_gpu as cu_grid;

        // load new entity data to gpu
        self.update_entities(entities);

        // update the grid with the new entity positions
        let overflow = self.add_to_grid();

        // update ligand positions
        unsafe{
            let delta_time = 1.0 / self.settings.fps();
            cu_grid::update_positions(self.ligands, self.ligand_count, delta_time);
        }
        
        // then, handle collisions
        let collisions;
        unsafe {
            let dim = Dim {
                x: self.settings.dimensions().0,
                y: self.settings.dimensions().1,
                depth: self.settings.cuda_slots_per_cell() as u32,
            };
            let receptor_count = self.settings.receptor_capacity() as u32;
            collisions = cu_grid::ligand_collision(search_radius, dim, self.grid, self.entities,
                self.ligands, self.ligand_count, self.receptors, receptor_count);
        }

        // clear grid
        let size = self.settings.dimensions().0 * self.settings.dimensions().1 * self.settings.cuda_slots_per_cell() as u32;
        unsafe {
            cu_grid::clear_grid(self.grid, size);
        }

        return (collisions, overflow);
    }
}