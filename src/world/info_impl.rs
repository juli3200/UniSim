use super::*;

impl World {
    /// Returns the current time in the simulation.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Returns the current population size of entities in the world.
    pub fn population_size(&self) -> usize {
        self.population_size
    }
    /// Returns the current count of ligands in the world.
    pub fn ligands_count(&self) -> usize {
        self.ligands_count
    }
    /// Returns the current counter value used for assigning unique IDs to new entities and ligands.
    pub fn counter(&self) -> usize {
        self.counter
    }
    
}

