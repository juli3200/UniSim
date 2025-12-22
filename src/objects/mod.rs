use ndarray::Array1;

pub mod entity;
pub mod ligand;
pub mod genome;
pub mod receptor;

pub(crate) const OUTPUTS: usize = 2; // number of different inner proteins / concentrations

#[derive(Debug, Clone)]
pub struct LigandSource {
    position: Array1<f32>,
    emission_rate: f32, // ligands per second
    ligand_spec: u16,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Ligand {
    pub(crate) emitted_id: usize, // id of the entity that emitted the ligand
    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) velocity: Array1<f32>, // velocity in the world
    pub(crate) spec: u16,
    pub(crate) energy: f32,
    //pub(crate) lifetime: u32, // lifetime in simulation steps remaining
}

#[derive(Debug, Clone)]
pub(crate) struct Genome{
    // outputs 
    pub(crate) move_threshold: i16, // threshold for movement decision
    pub(crate) ligand_emission_threshold: i16, // threshold for ligand emission decision
    pub(crate) ligands: Vec<u16>, //  types of ligands the entity can emit -> size: settings.ligand_types()

    // inputs
    pub(crate) receptor_dna: Vec<u64>, // DNA sequence for receptors -> size: settings.receptor_types()

}

#[derive(Debug, Clone)]
pub(crate) struct Entity {
    pub(crate) id: usize,

    // ******************** biological *********************

    // *********** GENOME *************
    pub(crate) genome: Genome, // GENOME of the entity


    pub(crate) energy: f32, // energy level of the entity
    pub(crate) age: usize, // age of the entity in simulation steps

    // ************ sensors and inner proteins ************

    // receptors
    pub(crate) receptors: Vec<u32>, // list of receptors (size: settings.receptor_capacity())

    // inner proteins
    // range defined in settings.concentration_range()
    pub(crate) inner_protein_levels: [i16; OUTPUTS], // concentration levels of different inner proteins
    

    // during update emit ligands
    ligands_to_emit: Vec<Ligand>, // ligands to emit
    pub(crate) received_ligands: Vec<u8>, // received ligands used for storing (angle from which ligand was received (in degrees from 0 - 180))

    // ******************* physics ***********************
    pub(crate) position: Array1<f32>, // position in the world
    pub(crate) size: f32, // size of the entity
    pub(crate) velocity: Array1<f32>, // velocity of the entity
    pub(crate) speed: f32, // current speed of the entity
    pub(crate) acceleration: Array1<f32>, // acceleration of the entity

    pub(crate) last_entity_collision: (usize, usize), // ids of the last collided entity and when it happened 
    pub(crate) last_border_collision: usize, // when the last border collision happened

    // ******************** cuda ************************
    #[cfg(feature = "cuda")]
    pub(crate) cuda_receptor_index: Option<u32>, // index of the entity receptors in the CUDA memory
}

