use super::*;

impl World {


    /// Returns the current time in the simulation.
    pub fn time(&self) -> f32 {
        self.time
    }
    /// Returns the current population size of entities in the world.
    pub fn population_size(&self) -> usize {
        self.entities.len()
    }
    /// Returns the current count of ligands in the world.
    pub fn ligands_count(&self) -> usize {
        self.ligands.len()
    }
}


pub fn get_entity<'a>(entities: &'a Vec<objects::Entity>, id: usize) -> Option<&'a objects::Entity> {
    entities.iter().find(|e| e.id == id)
}

pub fn get_entity_mut<'a>(entities: &'a mut Vec<objects::Entity>, id: usize) -> Option<&'a mut objects::Entity> {
    entities.iter_mut().find(|e| e.id == id)
}