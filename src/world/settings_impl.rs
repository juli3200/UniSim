use crate::world::Settings;


impl Settings {
    pub fn default(n: usize) -> Self {
        Self {
            default_population: n,
            dimensions: (100, 100),
            spawn_size: 1.0,
            give_start_vel: true,
            store_capacity: 1024,
            fps: 60.0,
            velocity: 3.0,
        }
    }


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

}