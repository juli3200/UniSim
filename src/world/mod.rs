use std::path::PathBuf;

use crate::objects;
use ndarray::Array1;
use crate::settings_::Settings;

mod info;
mod global;
mod space;
pub mod serialize;


/// a world is a 2D space filled with objects
/// e.g. ligands, entities, etc.
/// The interaction between these objects are handled in the world Struct
/// World provides all the functions to run the simulation
#[derive(Debug, Clone)]
pub struct World {
    // settings are stored in the settings struct
    // settings are used to configure the world
    pub settings: Settings,
    buffer: Vec<Vec<u8>>,  // Buffer for saving to reduce I/O time
    path: Option<PathBuf>,

    // variables
    init: bool,
    pub(crate) time: f32, // current time in the simulation

    pub(crate) counter: usize, // the counter is used to assign unique IDs to new entities and ligands
    pub(crate) byte_counter: usize, // used for jumper in save file
    pub(crate) iteration: usize, // number of iterations the store capacity has been increased

    // objects in the world
    pub(crate) entities: Vec<objects::Entity>,
    pub(crate) ligands: Vec<objects::Ligand>,
    pub(crate) ligand_sources: Vec<objects::LigandSource>,

    pub(crate) new_ligands: Vec<objects::Ligand>, // new ligands to be added to the world at the end of the step

    pub space: Space, // the space is used to store the entities and ligands in a 2D grid

    #[cfg(feature = "cuda")]
    pub cuda_world: Option<crate::cuda::CUDAWorld>, // the CUDA world is used to store the entities and ligands in a single array for CUDA processing


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
    EntityCollision(Array1<f32>, f32, Array1<f32>, usize), // velocity and mass of the colliding entity, position, id
}

#[derive(Debug, Clone)]
pub struct Space {
    pub settings: Settings, // needs to be updated with changes
    pub width: u32,
    pub height: u32,
    pub max_size: f32, // the biggest size of an entity in the world (used for efficient space checking)
    grid: Vec<Vec<Vec<usize>>>, // 2D grid of indices of entities ids
}
