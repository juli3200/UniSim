#![allow(non_snake_case)]
#![allow(dead_code)]
//! # Simulation of evolving unicellular organisms
//! ### Maturarbeit Julian Heer Jahrgang 25/26
//! This project aims to simulate the behavior and evolution of unicellular organisms
//! in a controlled environment, allowing for the study of their interactions and adaptations.


pub mod world;
mod objects;

#[cfg(feature = "cuda")]
mod cuda;
