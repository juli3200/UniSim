pub mod entity;
pub mod ligand;

#[derive(Debug, Clone)]
pub struct Ligand {
    pub id: usize,
}
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: usize,
}