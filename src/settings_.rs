#![allow(dead_code)]


use paste::paste;
use serde::Deserialize;

// doesn't actually provide any security, just a deterrent
const SECRET_KEY: u32 = 31425364; // key is used to ensure settings changes are only done through macros



macro_rules! get_set_maker{
    (struct $struct_name: ident{
        $(
            $field_name: ident : $field_type: ty $( { $check:ident } )?
        ),* $(,)?
    }) => {

        #[derive(Debug, Clone, Deserialize)]
        #[serde(default)]
        pub struct $struct_name {
            $($field_name : $field_type ),*
        }

        // implement getter functions
        impl $struct_name {
            $(
                pub(crate) fn $field_name(&self) -> $field_type {
                    self.$field_name.clone()
                }
            )*
        }

        // implement setter functions using paste to create set_<field>
        paste! {
            impl $struct_name {
                $(
                    pub fn [<set_ $field_name>](&mut self, value: $field_type, key: u32) {
                        if key != SECRET_KEY {
                            eprintln!("Only edit settings through macros");
                            return;
                        }
                        $( $check!(self, value, $field_name)  )?;
                        self.$field_name = value;
                    }
                )*
            }
        }
    };
}


macro_rules! reject_init {
    ($s:ident, $v:ident, $name:ident) => {
        if $s.init {
            eprintln!("Cannot change {} after initialization", stringify!($name));
            return;
        }
    };
}

macro_rules! std_dev {
    ($s:ident, $v:ident, $name:ident) => {
        if $v <= 0.0 {
            eprint!("Invalid {}: must be positive", stringify!($name));
            return;
        }
    };
}

macro_rules! conc_range {
    ($s:ident, $v:ident, $name:ident) => {
        if $s.init {
            eprint!("Cannot change {} after initialization", stringify!($name));
            return;
        }

        if $v.0 >= $v.1 {
            eprint!("Invalid {}: start must be less than end", stringify!($name));
            return;
        }

        // prevent overflow
        if $v.0 == i16::MIN || $v.1 == i16::MAX {
            eprint!("Invalid {}: values must be within i16 bounds", stringify!($name));
            return;
        }

        if $v.0 - $v.1 < 5 {
            eprint!("Warning: {} is very small, may lead to frequent clamping", stringify!($name));
        }
    };
}

macro_rules! prob {
    ($s:ident, $v:ident, $name:ident) => {
        if $v < 0.0 || $v > 1.0 {
            eprint!("Invalid {} probability", stringify!($name));
            return;
        }
    };
}

macro_rules! energy {
    ($s:ident, $v:ident, $name:ident) => {
        if $v <= 0.0 {
            eprint!("Invalid {}: must be positive", stringify!($name));
            return;
        }
        if $v > 10.0 {
            eprint!("Warning: {} is very high, may lead to instability", stringify!($name));
        }
    };
}
get_set_maker!(
struct Settings {
    init: bool {reject_init},

    // unchangeable settings
    default_population: usize {reject_init}, // default population size of entities
    dimensions: (u32, u32) {reject_init}, // width, height of the world
    spawn_size: f32 {reject_init}, // size of the entities when they are spawned
    store_capacity: usize {reject_init}, // capacity of the save file
    give_start_vel: bool {reject_init}, // whether to give entities a starting velocity

    concentration_range: (i16, i16) {conc_range}, // range of concentrations for inner proteins
    fps: f32 {reject_init}, // frames per second of the simulation

    receptor_capacity: usize {reject_init}, // number of total receptors per entity
    different_receptors: usize, // number of different receptor types
    different_ligands: usize, // number of different ligand types an entity can emit
    reproduction_threshold: Option<i16>, // reproduction threshold for entities
    reproduction_probability: f64 {prob}, // probability of reproduction when threshold is met

    standard_deviation: f64 {std_dev}, // standard deviation for random values
    mean: f64, // mean for random values


    // changeable settings
    velocity: f32, // default velocity of entities
    ligand_velocity: f32, // default velocity of ligands
    gravity: Vec<f32>, // gravity of the world
    drag: f32, // drag/friction of the world
    idle_energy_cost: f32, // energy cost per second depending on Area

    enable_entity_ligand_emission: bool,

    tumble_chance: f64 {prob}, // chance of tumbling
    max_energy_ligand: f32 {energy}, // maximum energy of a ligand
    movement_energy_cost: f32, // energy cost of movement
    ligand_emission_energy_cost: f32, // threshold for emitting ligands


    // cuda settings
    //#[cfg(feature = "cuda")]
    cuda_slots_per_cell: usize, // number of slots per cell in the cuda grid
});




impl Default for Settings {
    fn default() -> Self {
        Self::blueprint(100)
    }
}


impl Settings {
    // initalized settings
    pub fn new(n: usize) -> Self {
        let settings = Self::blueprint(n);
        settings
    }

    // uninitialized blueprint
    pub fn blueprint(n: usize) -> Self {
        Self {
            init: false,
            default_population: n,
            dimensions: (100, 100),
            spawn_size: 1.0,
            give_start_vel: true,
            concentration_range: (-32, 32),

            receptor_capacity: 10_000, // 10_000 receptors
            different_receptors: 10, // 10 different receptor types
            different_ligands: 10, // 10 different ligand types
            reproduction_threshold: None, // default reproduction threshold
            reproduction_probability: 0.5, // 50% chance of reproduction

            standard_deviation: 10.0,
            mean: 0.0,

            store_capacity: 1024,
            fps: 60.0,
            velocity: 3.0,
            ligand_velocity: 5.0,
            gravity: vec![0.0, 0.0],
            drag: 0.0,
            idle_energy_cost: 1e-3,

            enable_entity_ligand_emission: true,

            #[cfg(feature = "cuda")]
            cuda_slots_per_cell: 10,

            tumble_chance: 0.3333,
            max_energy_ligand: 1.0,
            movement_energy_cost: 1.0,
            ligand_emission_energy_cost: 0.001,
        }
    }

    pub fn initialize(&mut self) {
        self.init = true;
    }

}