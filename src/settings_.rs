#![allow(dead_code)]


use paste::paste;
use serde::Deserialize;

// doesn't actually provide any security, just a deterrent
const SECRET_KEY: u32 = 31425364; // key is used to ensure settings changes are only done through macros



macro_rules! get_set_maker{
    (struct $struct_name: ident{
        $(
            $field_name: ident : $field_type: ty $( { $($check:ident),* } )?
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
                pub fn $field_name(&self) -> $field_type {
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
                        $(
                            $($check!(self, value, $field_name);)*
                        )?
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

macro_rules! possible_ligands {
    ($s:ident, $v:ident, $name:ident) => {
        if $s.init {
            eprint!("Cannot change {} after initialization", stringify!($name));
            return;
        }

        if ($v as u32).count_ones() != 1 {
            eprint!("Invalid {}: must be a power of 2", stringify!($name));
            return;
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

        concentration_range: (i16, i16) {reject_init, conc_range}, // range of concentrations for inner proteins
        fps: f32 {reject_init}, // frames per second of the simulation

        receptor_capacity: usize {reject_init}, // number of total receptors per entity

        // mutable settings
        possible_ligands: usize {possible_ligands}, // number of possible ligands an entity can emit
        receptors_per_entity: u32, // number of different receptor types
        ligands_per_entity: u32, // number of different ligand types an entity can emit
        max_age: usize, // maximum age of an entity in simulation seconds (0 = infinite)
        max_size: f32, // maximum size of an entity

        mutation_rate: f64, // probability of mutation per receptor bit
        threshold_change: f64, // \sigma for threshold changes

        standard_deviation_threshold: f64 {std_dev}, // standard deviation for random values
        mean_threshold: f64, // mean for random values

        velocity: f32, // default velocity of entities
        ligand_velocity: f32, // default velocity of ligands
        gravity: (f32, f32), // gravity of the world
        drag: f32, // drag/friction of the world
        idle_energy_cost: f32, // energy cost per second depending on Area
        entity_run_energy_cost: f32, // energy cost per second when moving
        entity_tumble_energy_cost: f32, // energy cost per tumble
        ligand_poisoning_active: bool, // whether ligand poisoning is active

        enable_entity_ligand_emission: bool,

        tumble_chance: f64 {prob}, // chance of tumbling
        max_energy_ligand: f32 {energy}, // maximum energy of a ligand
        min_energy_ligand: f32 {energy}, // minimum energy of a ligand
        movement_energy_cost: f32, // energy cost of movement
        ligand_emission_energy_cost: f32, // threshold for emitting ligands

        standard_plasmid_count: u32, // number of standard plasmids per entity
        entity_acceleration: f32, // acceleration of entities


        // cuda settings
        cuda_slots_per_cell: usize, // number of slots per cell in the cuda grid
        cuda_memory_interval: usize, // interval for cuda memory reallocation
    }
);




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
            ligand_poisoning_active: false,

            receptor_capacity: 10_000, // 10_000 receptors
            receptors_per_entity: 2, // 2 different receptor types
            ligands_per_entity: 2, // 2 different ligand types it can emit
            max_age: 50, 
            max_size: 2.0,

            standard_deviation_threshold: 10.0,
            mean_threshold: 0.0,

            mutation_rate: 0.001,
            threshold_change: 1.5,

            store_capacity: 1024,
            fps: 60.0,
            velocity: 3.0,
            ligand_velocity: 5.0,
            gravity: (0.0, 0.0),
            drag: 0.0,
            idle_energy_cost: 1e-3,

            enable_entity_ligand_emission: true,
            standard_plasmid_count: 1,


            
            cuda_slots_per_cell: 10,
            cuda_memory_interval: 10000,

            tumble_chance: 0.3333,
            max_energy_ligand: 1.0,
            min_energy_ligand: 0.1,
            movement_energy_cost: 1.0,
            ligand_emission_energy_cost: 0.001,
            possible_ligands: 16, // must be a power of 2

            entity_acceleration: 1.0,
            entity_run_energy_cost: 0.001,
            entity_tumble_energy_cost: 0.005,
        }
    }

    pub fn initialize(&mut self) {
        self.init = true;
    }

}