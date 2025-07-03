pub mod entity;
pub mod ligand;

#[derive(Debug, Clone)]
pub struct Ligand {
    pub id: usize,
}
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: usize,
    pub position: (f32, f32), // position in the world
    pub size: f32, // size of the entity
}