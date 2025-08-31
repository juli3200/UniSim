use crate::world::Settings;


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
            store_capacity: 1024,
            fps: 60.0,
            velocity: 3.0,
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

    // returns frames per second
    pub fn fps(&self) -> f32 {
        self.fps
    }

    // returns default velocity
    pub fn velocity(&self) -> f32 {
        self.velocity
    }

    //
    //
    // setter fn
    //
    //

    pub fn set_default_population(&mut self, population: usize) {
        if self.init {
            panic!("Cannot change default population after initialization");
        }
        self.default_population = population;
    }

    pub fn set_dimensions(&mut self, dimensions: (u32, u32)) {
        if self.init {
            panic!("Cannot change dimensions after initialization");
        }
        self.dimensions = dimensions;
    }

    pub fn set_spawn_size(&mut self, spawn_size: f32) {
        if self.init {
            panic!("Cannot change spawn size after initialization");
        }
        self.spawn_size = spawn_size;
    }

    pub fn set_give_start_vel(&mut self, give_start_vel: bool) {
        if self.init {
            panic!("Cannot change give_start_vel after initialization");
        }
        self.give_start_vel = give_start_vel;
    }

    pub fn set_store_capacity(&mut self, store_capacity: usize) {
        if self.init {
            panic!("Cannot change store_capacity after initialization");
        }
        self.store_capacity = store_capacity;
    }


    pub fn set_fps(&mut self, fps: f32) {
        self.fps = fps;
    }

    pub fn set_velocity(&mut self, velocity: f32) {
        self.velocity = velocity;
    }

}