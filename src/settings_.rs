#![allow(dead_code)]

use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct Settings {
    init: bool, // whether the settings have been initialized

    // unchangeable settings
    default_population: usize, // default population size of entities
    dimensions: (u32, u32), // width, height of the world
    spawn_size: f32, // size of the entities when they are spawned
    store_capacity: usize, // capacity of the save file
    give_start_vel: bool, // whether to give entities a starting velocity
    concentration_range: (i16, i16), // range of concentrations for inner proteins
    receptor_capacity: usize, // number of receptors per entity


    // changeable settings
    fps: f32, // frames per second of the simulation
    velocity: f32, // default velocity of entities
    gravity: Array1<f32>, // gravity of the world
    friction: f32, // mü of the world
    tumble_chance: f64, // chance of tumbling

    // cuda settings
    #[cfg(feature = "cuda")]
    pub(crate) cuda_slots_per_cell: usize, // number of slots per cell in the cuda grid
}


impl Settings {
    // initalized settings
    pub fn new(n: usize) -> Self {
        let mut settings = Self::blueprint(n);
        settings.init();
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

            store_capacity: 1024,
            fps: 60.0,
            velocity: 3.0,
            gravity: Array1::zeros(2),
            friction: 0.0,
            #[cfg(feature = "cuda")]
            cuda_slots_per_cell: 10,
            tumble_chance: 0.3333,
        }
    }

    pub fn init(&mut self) {
        self.init = true;
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

    // returns frames per second
    pub fn fps(&self) -> f32 {
        self.fps 
    }

    // returns default velocity
    pub fn velocity(&self) -> f32 {
        self.velocity
    }

    // returns gravity
    pub fn gravity(&self) -> Array1<f32> {
        self.gravity.clone()
    }

    // returns friction
    pub fn friction(&self) -> f32 {
        self.friction
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_slots_per_cell(&self) -> usize {
        self.cuda_slots_per_cell
    }
    pub fn tumble_chance(&self) -> f64 {
        self.tumble_chance
    }

    //
    //
    // setter fn
    //
    //

    pub fn set_default_population(&mut self, population: usize) {
        if self.init {
            eprint!("Cannot change default population after initialization");
            return;
        }
        self.default_population = population;
    }

    pub fn set_dimensions(&mut self, dimensions: (u32, u32)) {
        if self.init {
            eprint!("Cannot change dimensions after initialization");
            return;
        }
        self.dimensions = dimensions;
    }

    pub fn set_spawn_size(&mut self, spawn_size: f32) {
        if self.init {
            eprint!("Cannot change spawn size after initialization");
            return;
        }
        self.spawn_size = spawn_size;
    }

    pub fn set_give_start_vel(&mut self, give_start_vel: bool) {
        if self.init {
            eprint!("Cannot change give_start_vel after initialization");
            return;
        }
        self.give_start_vel = give_start_vel;
    }

    pub fn set_store_capacity(&mut self, store_capacity: usize) {
        if self.init {
            eprint!("Cannot change store_capacity after initialization");
            return;
        }
        self.store_capacity = store_capacity;
    }

    pub fn set_concentration_range(&mut self, range: (i16, i16)) {
        if self.init {
            eprint!("Cannot change concentration_range after initialization");
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

    pub fn set_receptor_capacity(&mut self, capacity: usize) {
        if self.init {
            eprint!("Cannot change receptor_capacity after initialization");
            return;
        }
        self.receptor_capacity = capacity;
    }


    pub fn set_fps(&mut self, fps: f32) {
        self.fps = fps;
    }

    pub fn set_velocity(&mut self, velocity: f32) {
        self.velocity = velocity;
    }

    pub fn set_gravity<T: Into<Array1<f32>>>(&mut self, gravity: T) {
        self.gravity = gravity.into();
    }

    pub fn set_friction(&mut self, friction: f32) {
        self.friction = friction;
    }

    #[cfg(feature = "cuda")]
    pub fn set_cuda_slots_per_cell(&mut self, slots: usize) {
        self.cuda_slots_per_cell = slots;
    }

    pub fn set_tumble_chance(&mut self, chance: f64) {
        if chance < 0.0 || chance > 1.0 {
            eprint!("Tumble chance must be between 0.0 and 1.0");
            return;
        }
        self.tumble_chance = chance;
    }

}