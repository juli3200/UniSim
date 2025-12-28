//#![allow(dead_code)]


use paste::paste;
use serde::Deserialize;
use std::collections::HashMap;

// doesn't actually provide any security, just a deterrent
const SECRET_KEY: u32 = 31425364; // key is used to ensure settings changes are only done through macros



macro_rules! get_set_maker{
    (struct $struct_name: ident{
        $(
            $field_name: ident : $field_type: ty, $field_value: expr $(, { $($check:ident),* } )?
        ),* $(,)?
    }) => {

        #[derive(Debug, Clone, Deserialize)]
        #[serde(default)]
        pub struct $struct_name {
            $(
                $field_name : $field_type,
            )*

            #[serde(flatten, default)]
            pub extra: std::collections::HashMap<String, serde_json::Value>,
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

        impl Default for $struct_name {
            fn default() -> Self {
                Self {
                    $( $field_name: $field_value ),*,
                    extra: HashMap::new(),
                }
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
// RENAME EVERYTHING todo
get_set_maker!(
    struct Settings {
        init: bool, false, {reject_init},

        // unchangeable settings
        default_population: usize, 100, {reject_init}, // default population size of entities
        dimensions: (u32, u32), (100, 100), {reject_init}, // width, height of the world
        spawn_size: f32, 1.0, {reject_init}, // size of the entities when they are spawned
        store_capacity: usize, 1000, {reject_init}, // capacity of the save file
        give_start_vel: bool, true, {reject_init}, // whether to give entities a starting velocity
        concentration_range: (i16, i16), (-32, 32), {reject_init, conc_range}, // range of concentrations for inner proteins
        fps: f32, 60.0, {reject_init}, // frames per second of the simulation

        receptors_per_entity: usize, 10_000, {reject_init}, // number of total receptors per entity
        
        // mutable settings
        possible_ligands: usize, 16, {possible_ligands}, // number of possible ligands an entity can emit (must be a power of 2)
        receptor_types_per_entity: u32, 2, // number of different receptor types
        ligands_per_entity: u32, 2, // number of different ligand types an entity can emit
        max_age: usize, 50 ,// maximum age of an entity in simulation seconds (0 = infinite)
        max_size: f32, 2.0, // maximum size of an entity

        mutation_rate: f64, 0.001, // probability of mutation per receptor bit
        std_dev_mutation: f64, 1.5, {std_dev}, // \sigma for threshold changes
        std_dev_random: f64, 10.0, {std_dev}, // standard deviation for random values
        mean_random: f64, 0.0, // mean for random values

        max_plasmid_count: usize, 5, // maximum number of plasmids per entity
        standard_plasmid_count: usize, 1, // start count of plasmids per entity

        velocity: f32, 3.0, // default velocity of entities
        ligand_velocity: f32, 2.0, // default velocity of ligands
        general_force: (f32, f32), (0.0, 0.0), // gravity of the world
        drag: f32, 0.5, // drag/friction of the world
        idle_energy_cost: f32, 1e-3, // energy cost per second depending on Area
        entity_acceleration: f32, 1.0, 
        entity_run_energy_cost: f32, 0.001, // energy cost of entity movement
        entity_tumble_energy_cost: f32, 0.005, // energy cost of entity tumbling

        enable_entity_ligand_emission: bool, true, // whether entities can emit ligands


        tumble_chance: f64, 0.3333,{prob},  // chance of tumbling
        toxins_active: bool, true, // whether ligand poisoning is active
        max_energy_ligand: f32, 1.0, {energy}, // maximum energy of a ligand
        min_energy_ligand: f32, 0.1, {energy}, // minimum energy of a ligand
        //ligand_emission_energy_cost: f32, 0.001, // energy cost per ligand emitted

        // cuda settings
        cuda_slots_per_cell: usize, 10, // number of slots per cell in the cuda grid
        //cuda_memory_interval: usize, 10000, // interval for cuda memory reallocation

        path: String, "saves/sim".to_string(), // default save path


    }
);







impl Settings {
    // initalized settings
    pub fn new() -> Self {
        let mut settings = Self::default();
        settings.initialize();
        settings
    }

    pub fn initialize(&mut self) {
        self.init = true;
    }

}