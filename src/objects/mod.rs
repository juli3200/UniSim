use ndarray::Array1;

pub mod entity;
pub mod ligand;
mod receptor;

#[derive(Debug, Clone)]
pub(crate) enum ObjectType {
    Entity(usize),
    Ligand,
}

#[derive(Debug, Clone)]
pub(crate) struct Ligand {
    pub(crate) emitted_id: usize, // id of the entity that emitted the ligand
    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) velocity: Array1<f32>, // velocity in the world
    pub(crate) message: u32 // message carried by the ligand
}
#[derive(Debug, Clone)]
pub(crate) struct Entity {
    pub(crate) id: usize,

    // ******************** biological *********************
    energy: f32, // energy level of the entity
    dna: Vec<u128>, // DNA sequence of the entity
    age: usize, // age of the entity in simulation steps
    reproduction_rate: f32, // rate of reproduction

    // receptors
    receptors: Vec<u32>, // list of receptor functions
    //...
    // concentrations


    // ******************* physics ***********************
    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) size: f32, // size of the entity
    pub(crate) velocity: Array1<f32>, // velocity of the entity
    pub(crate) acceleration: Array1<f32>, // acceleration of the entity

    pub(crate) last_entity_collision: (usize, usize), // ids of the last collided entity and when it happened 
    pub(crate) last_border_collision: usize, // when the last border collision happened
}

struct ReceptorBond {
    energy_change: f32, 
    concentration_type: u8, // which concentration to change
    concentration_change: f32, // how much to change the concentration
}