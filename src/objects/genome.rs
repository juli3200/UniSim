use crate::prelude::Settings;
use crate::objects::receptor::sequence_receptor;
use super::Genome;
use rand::Rng;
use rand_distr::{Distribution, Normal};

impl Genome {
    #[allow(dead_code)]
    pub fn new(move_threshold: i16, ligand_emission_threshold: i16, ligands: Vec<u16>, receptor_dna: Vec<u64>) -> Self {
        Self {
            move_threshold,
            ligand_emission_threshold,
            ligands,
            receptor_dna,
        }
    }

    pub fn mutate(&self, settings: &crate::settings_::Settings) -> Self {

        let mut rng = rand::rng();

        let mut new_genome = self.clone();


        // change MUTATION MODE !!!!!!!!!!
        // todo

        new_genome.move_threshold += rng.random_range(-2..=2);
        new_genome.ligand_emission_threshold += rng.random_range(-2..=2);

        // mutate ligands
        // only allow valid bit flips
        for ligand in &mut new_genome.ligands {
            let i = settings.ligands_per_entity().count_ones();
            for j in 0..i {
                if rng.random_bool(settings.mutation_rate()) {
                    // flip bit j
                    *ligand ^= 1 << j;
                }
            }
        }

        // mutate receptor DNA
        for receptor in &mut new_genome.receptor_dna {
            // mutate each bit with probability mutation_rate
            // ensure resulting receptor is valid
            // if not do it again

            let mut c = 0;

            *receptor = loop {
                let mut mutate_rec = *receptor;

                for i in 0..64 {
                    if rng.random_bool(settings.mutation_rate()) {
                        mutate_rec ^= 1 << i; // flip bit i
                    }
                }

                if valid_rec_gene(*receptor, settings) {
                    break mutate_rec;
                }

                c += 1;
                if c > 1000 {
                    //eprintln!("Warning: could not mutate receptor gene to valid state after 1000 tries");
                    break *receptor; // give up and return original
                }
            };

            if rng.random_bool(settings.mutation_rate() * 64.0) {
                *receptor ^= 1 << rng.random_range(0..64); // flip a random bit

                // Ensure spec (bits 48-63) is still in range
                let spec_mask = 0xFFFFu64 << 48;
                let spec = ((*receptor & spec_mask) >> 48) as u16;
                let max_spec = settings.possible_ligands() as u16;
                if spec > max_spec {
                    *receptor = (*receptor & !spec_mask) | ((max_spec as u64) << 48);
                }

                // ensure what (bits 32-39) is still in range
                // random val from 0 to OUTPUTS-1
                
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


        let ligands: Vec<u16> = (0..settings.ligands_per_entity())
            .map(|_| random_ligand(settings))
            .collect();

        let receptor_dna: Vec<u64> = (0..settings.receptors_per_entity())
            .map(|_| random_receptor_genome(settings))
            .collect();

        Self {
            move_threshold,
            ligand_emission_threshold,
            ligands,
            receptor_dna,
        }
    }
}



fn random_receptor_genome(settings: &Settings) -> u64 {
    let mut rng = rand::rng();

    let info = 0u8;
    let a = rng.random_range(0..=u8::MAX);
    let b = rng.random_range(0..=u8::MAX);
    let c = rng.random_range(0..=u8::MAX);

    let what = rng.random_range(0..super::OUTPUTS as u8); // which inner protein does this receptor bind to
    let how_mutch: u8 = if rand::random_bool(0.5) { 1 } else { 0 }; // does it increase or decrease the concentration of the inner protein
    let spec = rng.random_range(0..=settings.possible_ligands()); // specificity of the receptor

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

fn random_ligand(settings: &Settings) -> u16{
    let mut rng = rand::rng();

    let spec = rng.random_range(0..=settings.possible_ligands() as u16);
    spec
}

fn valid_rec_gene(gene: u64, settings: &Settings) -> bool {
    let gene = u32::from_le_bytes(gene.to_le_bytes()[4..8].try_into().unwrap());

    let (inner_protein, _, receptor_spec) = sequence_receptor(gene);
    if inner_protein as usize >= super::OUTPUTS {
        return false;
    }

    let max_spec = settings.possible_ligands() as u16;
    if receptor_spec > max_spec {
        return false;
    }

    true
}
