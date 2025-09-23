use ndarray::Array1;

pub mod entity;
pub mod ligand;

#[derive(Debug, Clone)]
pub(crate) enum ObjectType {
    Entity(usize),
    Ligand(usize),
}

#[derive(Debug, Clone)]
pub(crate) struct Ligand {
    pub emitted_id: usize, // id of the entity that emitted the ligand
    pub position: Array1<f32>, // position in the world
    pub velocity: Array1<f32>, // velocity in the world
    pub message: u32 // message carried by the ligand
}
#[derive(Debug, Clone)]
pub(crate) struct Entity {
    pub(crate) id: usize,

    // biological
    pub(crate) energy: f32, // energy level of the entity
    pub(crate) dna: Vec<u32>, // DNA sequence of the entity
    pub(crate) age: usize, // age of the entity in simulation steps
    pub(crate) reproduction_rate: f32, // rate of reproduction
    //...



    // physics
    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) size: f32, // size of the entity
    pub(crate) velocity: Array1<f32>, // velocity of the entity
    pub(crate) acceleration: Array1<f32>, // acceleration of the entity

    pub(crate) last_collision: Option<(usize, usize)>, // id of the last collided entity and when it happened

}