#![allow(non_snake_case)]
//#[allow(dead_code)]


//! # Simulation of evolving unicellular organisms
//! ### Maturarbeit Julian Heer Jahrgang 25/26
//! This project aims to simulate the behavior and evolution of unicellular organisms
//! in a controlled environment, allowing for the study of their interactions and adaptations.


pub mod prelude;
mod world;
mod objects;
mod test;
mod settings_;

#[cfg(feature = "cuda")]
mod cuda;

// ******************************************************** MACROS ***********************************************************


// the settings macros are used for an easier setup 

// use feature: macro_metavar_expr_concat to use the macros
// if not just not use the macros

// this macro is used to edit the settings of the world AND the space
// it also can only change changeable settings it panics
#[allow(dead_code)]
#[macro_export]
macro_rules!  edit_settings {
    ($world:expr, $($setting:ident = $value:expr),+) => {
        // concat puts "set_" and the identifier together
        // so e.g. "set_fps"
        // then calls the setter function to change the value
        // via the macro syntax(...)+ it can take multiple settings
        paste! {
            $( ($world).settings.[<set_ $setting>]($value); )+
        }

        paste! {
            $( ($world).space.settings.[<set_ $setting>]($value); )+
        }
        #[cfg(feature = "cuda")]
        {
            if $world.cuda_world.is_some() {
                paste! {
                    $(($world).cuda_world.as_mut().unwrap().settings.[<set_ $setting>]($value);)+
                }
            }
        }
    };
}

// Macro to create settings
// declared here that it can access private fields
#[allow(dead_code)]
#[macro_export]
macro_rules! settings {
    () => {
        crate::Settings::new(100)
    };
    ( $n:expr ) => {
        Settings::new($n)
    };
    ($($setting:ident = $value:expr),+ $(,)?) => {
        {
            let mut settings = Settings::blueprint(100);
            paste! {
                $(
                    settings.[<set_ $setting>]($value);
                )+
            }
            settings.init();
            settings
        }
    };
    ($n:expr, $($setting:ident = $value:expr),+ $(,)?) => {
        {
            let mut settings = Settings::blueprint($n);
            paste! {
                $(
                    settings.[<set_ $setting>]($value);
                )+
            }
            settings.init();
            settings
        }
    };
}


