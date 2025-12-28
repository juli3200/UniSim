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
        fn check_type(_w: &mut World){}
        check_type($world); // ensure that the first argument is a World

        let key = 31425364; // same as in settings.rs
        // key is used to ensure settings changes are only done through macros

        // paste puts "set_" and the identifier together
        // so e.g. "set_fps"
        // then calls the setter function to change the value
        // via the macro syntax(...)+ it can take multiple settings
        paste! {
            $( ($world).settings.[<set_ $setting>]($value, key); )+
        }

        paste! {
            $( ($world).space.settings.[<set_ $setting>]($value, key); )+
        }
        #[cfg(feature = "cuda")]
        {
            if $world.cuda_world.is_some() {
                paste! {
                    $(($world).cuda_world.as_mut().unwrap().settings.[<set_ $setting>]($value, key);)+
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
        Settings::new(100)
    };
    ($($setting:ident = $value:expr),+ $(,)?) => {
        {
            let key = 31425364; // same as in settings.rs
            // key is used to ensure settings changes are only done through macros
            let mut settings = Settings::default();
            paste! {
                $(
                    settings.[<set_ $setting>]($value, key);
                )+
            }
            settings
        }
    };
    ( $path:expr ) => {
        {

            fn open_json<S>(path: S) -> std::io::Result<Settings> 
                where S: AsRef<std::path::Path>   
                {

                let file = std::fs::read_to_string(path)?;
                let settings: Settings = serde_json::from_str(&file)?;
                
                for (key, value) in settings.extra.iter() {
                    eprintln!("Warning: Unrecognized setting '{}' in settings file", key);
                }

                Ok(settings)
            }

            let settings = match open_json($path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to open settings file: {}", e);
                    Settings::new()
                }
            };
            settings
        }

    };

}


