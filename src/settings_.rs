#![allow(dead_code)]

use serde::Deserialize;

// doesn't actually provide any security, just a deterrent
const SECRET_KEY: u32 = 31425364; // key is used to ensure settings changes are only done through macros

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Settings {
    init: bool, // whether the settings have been initialized

    // unchangeable settings
    default_population: usize, // default population size of entities
    dimensions: (u32, u32), // width, height of the world
    spawn_size: f32, // size of the entities when they are spawned
    store_capacity: usize, // capacity of the save file
    give_start_vel: bool, // whether to give entities a starting velocity

    concentration_range: (i16, i16), // range of concentrations for inner proteins

    receptor_capacity: usize, // number of total receptors per entity
    different_receptors: usize, // number of different receptor types
    different_ligands: usize, // number of different ligand types an entity can emit

    standard_deviation: f64, // standard deviation for random values
    mean: f64, // mean for random values


    // changeable settings
    fps: f32, // frames per second of the simulation
    velocity: f32, // default velocity of entities
    ligand_velocity: f32, // default velocity of ligands
    gravity: Vec<f32>, // gravity of the world
    drag: f32, // drag/friction of the world

    enable_entity_ligand_emission: bool,

    tumble_chance: f64, // chance of tumbling
    max_energy_ligand: f32, // maximum energy of a ligand
    movement_energy_cost: f32, // energy cost of movement
    ligand_emission_energy_cost: f32, // threshold for emitting ligands


    // cuda settings
    #[cfg(feature = "cuda")]
    pub(crate) cuda_slots_per_cell: usize, // number of slots per cell in the cuda grid
}

impl Default for Settings {
    fn default() -> Self {
        Self::blueprint(100)
    }
}


impl Settings {
    // initalized settings
    pub fn new(n: usize) -> Self {
        let settings = Self::blueprint(n);
        settings
    }

    // uninitialized blueprint
    pub fn blueprint(n: usize) -> Self {
        Self {
            init: false,
            default_population: n,
            dimensions: (100, 100),
            spawn_size: 1.0,
            give_start_vel: true,
            concentration_range: (-32, 32),

            receptor_capacity: 10_000, // 10_000 receptors
            different_receptors: 10, // 10 different receptor types
            different_ligands: 10, // 10 different ligand types

            standard_deviation: 10.0,
            mean: 0.0,

            store_capacity: 1024,
            fps: 60.0,
            velocity: 3.0,
            ligand_velocity: 5.0,
            gravity: vec![0.0, 0.0],
            drag: 0.0,

            enable_entity_ligand_emission: true,

            #[cfg(feature = "cuda")]
            cuda_slots_per_cell: 10,

            tumble_chance: 0.3333,
            max_energy_ligand: 1.0,
            movement_energy_cost: 1.0,
            ligand_emission_energy_cost: 0.001,
        }
    }

    pub fn init(&mut self) {
        self.init = true;
    }

    pub fn is_init(&self) -> bool {
        self.init
    }

    //
    //
    // getter fn
    //
    //

    // returns default population
    pub fn default_population(&self) -> usize {
        self.default_population
    }

    // returns dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    // returns spawn size
    pub fn spawn_size(&self) -> f32 {
        self.spawn_size
    }

    // returns whether to give starting velocity
    pub fn give_start_vel(&self) -> bool {
        self.give_start_vel
    }

    // returns store capacity
    pub fn store_capacity(&self) -> usize {
        self.store_capacity
    }

    // returns concentration range
    pub fn concentration_range(&self) -> (i16, i16) {
        self.concentration_range
    }

    // returns receptor capacity
    pub fn receptor_capacity(&self) -> usize {
        self.receptor_capacity
    }

    // returns number of different receptor types
    pub fn different_receptors(&self) -> usize {
        self.different_receptors
    }

    // returns number of different ligand types
    pub fn ligand_types(&self) -> usize {
        self.different_ligands
    }

    // returns standard deviation
    pub fn standard_deviation(&self) -> f64 {
        self.standard_deviation
    }

    // returns mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    // returns frames per second
    pub fn fps(&self) -> f32 {
        self.fps 
    }

    // returns default velocity
    pub fn velocity(&self) -> f32 {
        self.velocity
    }

    // returns gravity
    pub fn gravity(&self) -> Vec<f32> {
        self.gravity.clone()
    }

    // returns drag
    pub fn drag(&self) -> f32 {
        self.drag
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_slots_per_cell(&self) -> usize {
        self.cuda_slots_per_cell
    }
    pub fn tumble_chance(&self) -> f64 {
        self.tumble_chance
    }

    pub fn max_energy_ligand(&self) -> f32 {
        self.max_energy_ligand
    }

    pub fn enable_entity_ligand_emission(&self) -> bool{
        self.enable_entity_ligand_emission
    }

    pub fn ligand_velocity(&self) -> f32 {
        self.ligand_velocity
    }
    //
    //
    // setter fn
    //
    //

    pub fn set_default_population(&mut self, population: usize, key: u32) {
        if self.init {
            eprint!("Cannot change default population after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.default_population = population;
    }

    pub fn set_dimensions(&mut self, dimensions: (u32, u32), key: u32) {
        if self.init {
            eprint!("Cannot change dimensions after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.dimensions = dimensions;
    }

    pub fn set_spawn_size(&mut self, spawn_size: f32, key: u32) {
        if self.init {
            eprint!("Cannot change spawn size after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.spawn_size = spawn_size;
    }

    pub fn set_give_start_vel(&mut self, give_start_vel: bool, key: u32) {
        if self.init {
            eprint!("Cannot change give_start_vel after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.give_start_vel = give_start_vel;
    }

    pub fn set_store_capacity(&mut self, store_capacity: usize, key: u32) {
        if self.init {
            eprint!("Cannot change store_capacity after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.store_capacity = store_capacity;
    }

    pub fn set_concentration_range(&mut self, range: (i16, i16), key: u32) {
        if self.init {
            eprint!("Cannot change concentration_range after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        if range.0 >= range.1 {
            eprint!("Invalid concentration range: start must be less than end");
            return;
        }

        // prevent overflow
        if range.0 == i16::MIN || range.1 == i16::MAX {
            eprint!("Invalid concentration range: values must be within i16 bounds");
            return;
        }

        if range.0 - range.1 < 5 {
            eprint!("Warning: concentration range is very small, may lead to frequent clamping");
        }

        self.concentration_range = range;
    }

    pub fn set_receptor_capacity(&mut self, capacity: usize, key: u32) {
        if self.init {
            eprint!("Cannot change receptor_capacity after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.receptor_capacity = capacity;
    }

    pub fn set_different_receptors(&mut self, different: usize, key: u32) {
        if self.init {
            eprint!("Cannot change different_receptors after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.different_receptors = different;
    }

    pub fn set_different_ligands(&mut self, different: usize, key: u32) {
        if self.init {
            eprint!("Cannot change different_ligands after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.different_ligands = different;
    }
    
    pub fn set_standard_deviation(&mut self, std_dev: f64, key: u32) {
        if self.init {
            eprint!("Cannot change standard_deviation after initialization");
            return;
        }
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        if std_dev <= 0.0 {
            eprint!("Standard deviation must be positive");
            return;
        }
        self.standard_deviation = std_dev;
    }

        pub fn set_mean(&mut self, mean: f64, key: u32) {
            if self.init {
                eprint!("Cannot change mean after initialization");
                return;
            }
            if key != SECRET_KEY {
                eprint!("Only edit settings through macros");
                return;
            }
            self.mean = mean;
        }


    pub fn set_fps(&mut self, fps: f32, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.fps = fps;
    }

    pub fn set_velocity(&mut self, velocity: f32, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.velocity = velocity;
    }

    pub fn set_gravity(&mut self, gravity: Vec<f32>, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.gravity = gravity;
    }

    pub fn set_drag(&mut self, drag: f32, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.drag = drag;
    }

    #[cfg(feature = "cuda")]
    pub fn set_cuda_slots_per_cell(&mut self, slots: usize, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.cuda_slots_per_cell = slots;
    }

    pub fn set_tumble_chance(&mut self, chance: f64, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        if chance < 0.0 || chance > 1.0 {
            eprint!("Tumble chance must be between 0.0 and 1.0");
            return;
        }
        self.tumble_chance = chance;
    }

    pub fn set_max_energy_ligand(&mut self, energy: f32, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        if energy <= 0.0 {
            eprint!("Max energy of ligand must be positive");
            return;
        }
        if energy > 10.0 {
            eprint!("Warning: max energy of ligand is very high, may lead to instability");
        }
        self.max_energy_ligand = energy;
    }

    pub fn set_enable_entity_ligand_emmision(&mut self, enable: bool, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.enable_entity_ligand_emission = enable;
    }

    pub fn set_ligand_velocity(&mut self, velocity: f32, key: u32) {
        if key != SECRET_KEY {
            eprint!("Only edit settings through macros");
            return;
        }
        self.ligand_velocity = velocity;
    }

}

