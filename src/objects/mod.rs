use ndarray::Array1;

pub mod entity;
pub mod ligand;
pub mod genome;
mod receptor;

const OUTPUTS: usize = 10; // number of different inner proteins / concentrations


#[derive(Debug, Clone)]
pub(crate) enum ObjectType {
    Entity(usize),
    Ligand,
}

#[derive(Debug, Clone)]
pub struct LigandSource {
    position: Array1<f32>,
    emission_rate: f32, // ligands per second
    ligand_message: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct Ligand {
    pub(crate) emitted_id: usize, // id of the entity that emitted the ligand
    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) velocity: Array1<f32>, // velocity in the world
    pub(crate) message: u32 // message carried by the ligand
}

#[derive(Debug, Clone)]
pub(crate) struct Genome{
    // outputs 
    move_threshold: i16, // threshold for movement decision
    ligand_emission_threshold: i16, // threshold for ligand emission decision
    ligands: Vec<u32>, // types of ligands the entity can emit -> size: settings.ligand_types()
    reproduction_threshold: i16, // threshold for reproduction decision, can be handled by a global setting (settings.reproduction_threshold() -> Option<i16>)

    // inputs
    receptor_dna: Vec<u64>, // DNA sequence for receptors -> size: settings.receptor_types()

}

#[derive(Debug, Clone)]
pub(crate) struct Entity {
    pub(crate) id: usize,

    // ******************** biological *********************

    // *********** GENOME *************
    pub(crate) genome: Genome, // GENOME of the entity


    energy: f32, // energy level of the entity
    age: usize, // age of the entity in simulation steps

    // ************ sensors and inner proteins ************

    // receptors
    receptors: Vec<u32>, // list of receptors (size: settings.receptor_capacity())

    // inner proteins
    // range defined in settings.concentration_range()
    pub(crate) inner_protein_levels: [i16; OUTPUTS], // concentration levels of different inner proteins
    

    // during update emit ligands
    ligands_to_emit: Vec<Ligand>, // ligands to emit

    // ******************* physics ***********************
    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) size: f32, // size of the entity
    pub(crate) velocity: Array1<f32>, // velocity of the entity
    pub(crate) acceleration: Array1<f32>, // acceleration of the entity

    pub(crate) last_entity_collision: (usize, usize), // ids of the last collided entity and when it happened 
    pub(crate) last_border_collision: usize, // when the last border collision happened
}

