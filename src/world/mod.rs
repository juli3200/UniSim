use std::path::PathBuf;

use crate::objects;
use ndarray::Array1;

mod info;
mod global;
mod space_impl;
mod settings_impl;
pub mod serialize;


/// a world is a 2D space filled with objects
/// e.g. ligands, entities, etc.
/// The interaction between these objects are handled in the world Struct
/// World provides all the functions to run the simulation
#[derive(Debug, Clone)]
pub struct World {
    // settings are stored in the settings struct
    // settings are used to configure the world
    pub(crate) settings: Settings,
    buffer: Vec<Vec<u8>>,  // Buffer for saving to reduce I/O time
    path: Option<PathBuf>,

    // variables
    pub(crate) time: f32, // current time in the simulation
    pub(crate) population_size: usize, // current population size of entities
    pub(crate) ligands_count: usize, // current count of ligands in the world

    pub(crate) counter: usize, // the counter is used to assign unique IDs to new entities and ligands
    pub(crate) byte_counter: usize, // used for jumper in save file
    pub(crate) saved_states: usize, // number of saved states in the save file
    pub(crate) iteration: usize, // number of iterations the store capacity has been increased

    // objects in the world
    pub(crate) entities: Vec<objects::Entity>,
    pub(crate) ligands: Vec<objects::Ligand>,

    pub(crate) space: Space, // the space is used to store the entities and ligands in a 2D grid

    #[cfg(feature = "cuda")]
    pub(crate) cuda_world: Option<crate::cuda::CUDAWorld>, // the CUDA world is used to store the entities and ligands in a single array for CUDA processing


}

#[derive(Debug, Clone)]
pub struct Settings {
    init: bool, // whether the settings have been initialized

    // unchangeable settings
    default_population: usize, // default population size of entities
    dimensions: (u32, u32), // width, height of the world
    spawn_size: f32, // size of the entities when they are spawned
    store_capacity: usize, // capacity of the save file
    give_start_vel: bool, // whether to give entities a starting velocity


    // changeable settings
    fps: f32, // frames per second of the simulation
    velocity: f32, // default velocity of entities
}

#[derive(Debug, Clone)]
pub(crate) enum SaveSlot{
    Jumper(usize),
    State(usize),
    None
}

#[derive(Debug, Clone)]
pub(crate) enum Border {
    Top,
    Bottom,
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub(crate) enum Collision {
    NoCollision,
    BorderCollision(Border),
    EntityCollision(Array1<f32>, f32, Array1<f32>), // velocity and mass of the colliding entity, position
}

#[derive(Debug, Clone)]
pub struct Space {
    pub settings: Settings, // needs to be updated with changes
    pub width: u32,
    pub height: u32,
    max_size: f32, // the biggest size of an entity in the world (used for efficient space checking)
    grid: Vec<Vec<Vec<objects::ObjectType>>>, // 2D grid of indices of entities and ligands
}
