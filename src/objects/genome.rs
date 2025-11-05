use crate::prelude::Settings;

use super::Genome;
use rand::Rng;
use rand_distr::{Distribution, Normal};

impl Genome {
    #[allow(dead_code)]
    pub fn new(move_threshold: i16, ligand_emission_threshold: i16, ligands: Vec<(f32, u16)>, reproduction_threshold: i16, receptor_dna: Vec<u64>) -> Self {
        Self {
            move_threshold,
            ligand_emission_threshold,
            ligands,
            reproduction_threshold,
            receptor_dna,
        }
    }

    pub fn mutate(&self) -> Self {
        let mut rng = rand::rng();

        let mut new_genome = self.clone();

        // change MUTATION MODE !!!!!!!!!!
        // todo

        new_genome.move_threshold += rng.random_range(-2..=2);
        new_genome.ligand_emission_threshold += rng.random_range(-2..=2);
        new_genome.reproduction_threshold += rng.random_range(-2..=2);

        // mutate ligands
        for ligand in &mut new_genome.ligands {
            if rng.random_bool(0.1) {
                ligand.0 = (ligand.0 + rng.random_range(-0.5..=0.5)).max(0.1); // mutate energy
            
            }
            if rng.random_bool(0.05) {
                ligand.1 = rng.random_range(0..=u16::MAX); // mutate spec
            }
        }

        // mutate receptor DNA
        for receptor in &mut new_genome.receptor_dna {
            if rng.random_bool(0.1) {
                *receptor ^= 1 << rng.random_range(0..64); // flip a random bit
            }
        }

        new_genome
    }

    pub fn random(settings: &crate::settings_::Settings) -> Self {
        let mut rng = rand::rng();
        let normal: Normal<f64> = Normal::new(settings.mean(), settings.standard_deviation()).unwrap();


        let min = settings.concentration_range().0 as f64;
        let max = settings.concentration_range().1 as f64;


        // sample thresholds from normal distribution and clamp to valid range
        let move_threshold = normal.sample(&mut rng).round().clamp(min, max) as i16;
        let ligand_emission_threshold = normal.sample(&mut rng).round().clamp(min, max) as i16;
        let reproduction_threshold = normal.sample(&mut rng).round().clamp(min, max) as i16;

        let ligands: Vec<(f32, u16)> = (0..settings.different_ligands())
            .map(|_| random_ligand(settings))
            .collect();

        let receptor_dna: Vec<u64> = (0..settings.different_receptors())
            .map(|_| random_receptor_genome())
            .collect();

        Self {
            move_threshold,
            ligand_emission_threshold,
            ligands,
            reproduction_threshold,
            receptor_dna,
        }
    }
}



fn random_receptor_genome() -> u64 {
    let mut rng = rand::rng();

    let info = 0u8;
    let a = rng.random_range(0..=u8::MAX);
    let b = rng.random_range(0..=u8::MAX);
    let c = rng.random_range(0..=u8::MAX);

    let what = rng.random_range(0..super::OUTPUTS as u8); // which inner protein does this receptor bind to
    let how_mutch: u8 = if rand::random_bool(0.5) { 1 } else { 0 }; // does it increase or decrease the concentration of the inner protein
    let spec = rng.random_range(0..=u16::MAX); // specificity of the receptor

    // Pack all fields into a u64 in little-endian order
    // Layout: [info (8 bits)][a (8)][b (8)][c (8)][what (8)][how_mutch (8)][spec (16)]
    let mut value: u64 = 0;
    value |= info as u64;
    value |= (a as u64) << 8;
    value |= (b as u64) << 16;
    value |= (c as u64) << 24;
    value |= (what as u64) << 32;
    value |= (how_mutch as u64) << 40;
    value |= (spec as u64) << 48;
    value
}

fn random_ligand(settings: &Settings) -> (f32, u16){
    let mut rng = rand::rng();

    let energy: f32 = rng.random_range(0.0..settings.max_energy_ligand());

    let spec = rng.random_range(0..=u16::MAX);

    (energy, spec)


}
