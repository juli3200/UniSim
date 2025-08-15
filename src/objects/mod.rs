use std::cell::RefCell;
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
}
#[derive(Debug, Clone)]
pub(crate) struct Entity {
    pub(crate) id: usize,

    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) size: f32, // size of the entity
    
    pub(crate) velocity: Array1<f32>, // velocity of the entity

}