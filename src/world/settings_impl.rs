use crate::world::Settings;


impl Settings {
    pub fn default() -> Self {
        Self {
            default_population: 100,
            dimensions: (100, 100),
            spawn_size: 1.0,
            fps: 60.0,
        }
    }
}