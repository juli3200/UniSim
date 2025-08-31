#![allow(non_snake_case)]
#![allow(dead_code)]
//! # Simulation of evolving unicellular organisms
//! ### Maturarbeit Julian Heer Jahrgang 25/26
//! This project aims to simulate the behavior and evolution of unicellular organisms
//! in a controlled environment, allowing for the study of their interactions and adaptations.


pub mod world;
mod objects;
mod test;



// this macro is used to edit the settings of the world AND the space
// it also can only change changeable settings
#[macro_export]
macro_rules! edit_settings {
    (&mut $world:expr, $($setting:ident = $value:expr),+) => {
        $( $world.settings.$setting = $value; )+
        $( $world.space.settings.$setting = $value; )+
    };
}

#[cfg(feature = "cuda")]
mod cuda;

