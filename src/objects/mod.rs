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
    pub id: usize,
    pub position: Array1<f32>, // position in the world
    pub velocity: Array1<f32>, // velocity in the world
}
#[derive(Debug, Clone)]
pub(crate) struct Entity {
    pub(crate) id: usize,

    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) size: f32, // size of the entity
    
    pub(crate) velocity: Array1<f32>, // velocity of the entity

    pub(crate) last_collision: Option<(usize, usize)>, // id of the last collided entity and when it happened

}