use super::*;


impl World{
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            time: 0.0,
            population_size: 0,
            ligands_count: 0,
            counter: 0,
            entities: Vec::new(),
            ligands: Vec::new(),
        }
    }
}