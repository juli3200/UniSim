#![allow(non_snake_case)]
#![allow(dead_code)]
// feature used for the settings macros
#![feature(macro_metavar_expr_concat)]


//! # Simulation of evolving unicellular organisms
//! ### Maturarbeit Julian Heer Jahrgang 25/26
//! This project aims to simulate the behavior and evolution of unicellular organisms
//! in a controlled environment, allowing for the study of their interactions and adaptations.



pub mod world;
mod objects;
mod test;
mod settings_;

#[cfg(feature = "cuda")]
mod cuda;


pub use settings_::Settings;


// ******************************************************** MACROS ***********************************************************


// the settings macros are used for an easier setup 

// use feature: macro_metavar_expr_concat to use the macros
// if not just not use the macros

// this macro is used to edit the settings of the world AND the space
// it also can only change changeable settings it panics
#[macro_export]
macro_rules! edit_settings {
    ($world:expr, $($setting:ident = $value:expr),+) => {
        // concat puts "set_" and the identifier together
        // so e.g. "set_fps"
        // then calls the setter function to change the value
        // via the macro syntax(...)+ it can take multiple settings

        $( $world.settings.${concat(set_, $setting)}($value); )+
        $( $world.space.settings.${concat(set_, $setting)}($value); )+
        #[cfg(feature = "cuda")]
        {
            if $world.cuda_world.is_some() {
                $($world.cuda_world.as_mut().unwrap().settings.${concat(set_, $setting)}($value);)+
            }
        }
    };
}

// Macro to create settings
// declared here that it can access private fields
#[macro_export]
macro_rules! settings {
    () => {
        crate::Settings::new(100)
    };
    ( $n:expr ) => {
        Settings::new($n)
    };
    ($($setting:ident = $value:expr),+) => {

        // same logic as in the edit settings macro

        {
            let mut settings = Settings::blueprint(100);
            $( settings.${concat(set_, $setting)}($value); )+
            settings.init();
            settings
        }
    };
    ($n:expr, $($setting:ident = $value:expr),+) => {
        {
            
            use crate::Settings; // compiler got issues idk
            let mut settings = Settings::blueprint($n);
            $( settings.${concat(set_, $setting)}($value); )+
            settings.init();
            settings
        }
    };
}


