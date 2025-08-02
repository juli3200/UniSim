use crate::objects;
use ndarray::Array1;

mod info;
mod global;
mod space_impl;
mod settings_impl;


/// a world is a 2D space filled with objects
/// e.g. ligands, entities, etc.
/// The interaction between these objects are handled in the world Struct
/// World provides all the functions to run the simulation
#[derive(Debug, Clone)]
pub struct World {
    // settings are stored in the settings struct
    // settings are used to configure the world
    pub settings: Settings,

    // variables
    pub(crate) time: f32, // current time in the simulation
    pub(crate) population_size: usize, // current population size of entities
    pub(crate) ligands_count: usize, // current count of ligands in the world

    pub(crate) counter: usize, // the counter is used to assign unique IDs to new entities and ligands

    // objects in the world
    pub(crate) entities: Vec<objects::Entity>,
    pub(crate) ligands: Vec<objects::Ligand>,

    pub(crate) space: Space, // the space is used to store the entities and ligands in a 2D grid


}

#[derive(Debug, Clone)]
pub struct Settings {
    // unchangeable settings
    pub default_population: usize, // default population size of entities 
    pub dimensions: (u32, u32), // width, height of the world
    pub spawn_size: f32, // size of the entities when they are spawned


    // changeable settings
    

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
    pub width: u32,
    pub height: u32,
    max_size: f32, // the biggest size of an entity in the world (used for efficient space checking)
    grid: Vec<Vec<Vec<objects::ObjectType>>>, // 2D grid of indices of entities and ligands
}
/* 
pub struct CUDAWorld {
    pub(crate) 
}*/