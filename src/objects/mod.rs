use std::cell::RefCell;

pub mod entity;
pub mod ligand;

#[derive(Debug, Clone)]
pub(crate) enum ObjectType {
    Entity(RefCell<Entity>),
    Ligand(RefCell<Ligand>),
}

#[derive(Debug, Clone)]
pub(crate) struct Ligand {
    pub id: usize,
}
#[derive(Debug, Clone)]
pub(crate) struct Entity {
    pub(crate) id: usize,
    pub(crate) position: (f32, f32), // position in the world
    pub(crate) size: f32, // size of the entity

    pub(crate) velocity: (f32, f32), // velocity of the entity
    
}