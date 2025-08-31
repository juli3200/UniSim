use crate::world::Settings;


impl Settings {
    pub fn default(n: usize) -> Self {
        Self {
            default_population: n,
            dimensions: (100, 100),
            spawn_size: 1.0,
            store_capacity: 1024,
            fps: 60.0,
            velocity: 3.0,
        }
    }
}