use crate::prelude::Settings;
use crate::objects::receptor::sequence_receptor;
use super::Genome;
use rand::Rng;
use rand_distr::{Distribution, Normal};

impl Genome {
    #[allow(dead_code)]
    pub fn new(move_threshold: i16, ligand_emission_threshold: i16, plasmid_threshold: i16, ligands: Vec<u16>, receptor_dna: Vec<u64>, toxins_active: bool, plasmids: Vec<u16>) -> Self {
        Self {
            move_threshold,
            ligand_emission_threshold,
            plasmid_threshold,
            ligands,
            receptor_dna,
            toxins_active,
            plasmids,
        }
    }

    pub fn mutate(&self, settings: &crate::settings_::Settings) -> Self {
        if settings.mutation_rate() >= 0.5{
            // high mutation rate means we just generate a new random genome
            return Genome::random(settings);
        }

        let mut rng = rand::rng();

        let mut new_genome = self.clone();


        // change MUTATION MODE !!!!!!!!!!
        // todo

        // mutate thresholds
        if settings.std_dev_mutation() == 0.0 && settings.mutation_rate() == 0.0  && settings.mean_random() == 0.0 {
            return new_genome;
        }
        let normal_dist = Normal::new(0.0, settings.std_dev_mutation()).unwrap();
        let move_threshold_delta = normal_dist.sample(&mut rng).round() as i16;
        let ligand_emission_threshold_delta = normal_dist.sample(&mut rng).round() as i16;
        let plasmid_threshold_delta = normal_dist.sample(&mut rng).round() as i16;

        new_genome.move_threshold += move_threshold_delta;
        new_genome.ligand_emission_threshold += ligand_emission_threshold_delta;
        new_genome.plasmid_threshold += plasmid_threshold_delta;

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
        for receptor in new_genome.receptor_dna.iter_mut() {
            // mutate each bit with probability mutation_rate
            // ensure resulting receptor is valid
            // if not do it again

            let mut c = 0;

            let mut mutate_rec = *receptor;

            let mut success: bool = true;

            // mutate every bit one by one
            // if after flipping a bit the receptor is invalid, try flipping it again

            for i in 0..64 {
                if !success {
                    // this is if c was too high last time
                    break;
                }

                success = loop {

                    if rng.random_bool(settings.mutation_rate()) {
                        mutate_rec ^= 1 << i; // flip bit i
                    }
                    if valid_rec_gene(mutate_rec, settings) {
                        break true;
                    }
                    c += 1;
                    if c > 10000 {
                        eprintln!("Warning: could not mutate receptor gene to valid state after 10000 tries");
                        break false; // give up and return original
                    }
                }
            }
            if success {
                *receptor = mutate_rec;
            }
        }



        new_genome
    }

    pub fn random(settings: &crate::settings_::Settings) -> Self {
        let mut rng = rand::rng();
        let normal: Normal<f64> = Normal::new(settings.mean_random(), settings.std_dev_random()).unwrap();


        let min = settings.concentration_range().0 as f64;
        let max = settings.concentration_range().1 as f64;


        // sample thresholds from normal distribution and clamp to valid range
        let move_threshold = normal.sample(&mut rng).round().clamp(min, max) as i16;
        let ligand_emission_threshold = normal.sample(&mut rng).round().clamp(min, max) as i16;
        let plasmid_threshold = normal.sample(&mut rng).round().clamp(min, max) as i16;


        let ligands: Vec<u16> = (0..settings.ligands_per_entity())
            .map(|_| random_ligand(settings))
            .collect();

        let receptor_dna: Vec<u64> = (0..settings.receptor_types_per_entity())
            .map(|_| random_receptor_genome(settings))
            .collect();

        let plasmids: Vec<u16> = (0..settings.standard_plasmid_count())
            .map(|_| random_plasmid_gene(settings))
            .collect();

        Self {
            move_threshold,
            ligand_emission_threshold,
            plasmid_threshold,
            ligands,
            receptor_dna,
            toxins_active: settings.toxins_active(),
            plasmids,
        }
    }
}


fn random_plasmid_gene(settings: &Settings) -> u16 {
    let mut rng = rand::rng();

    let spec = rng.random_range(0..settings.possible_ligands() as u16);
    spec
}


fn random_receptor_genome(settings: &Settings) -> u64 {
    let mut rng = rand::rng();

    let info = 0u8;
    let a = rng.random_range(0..=u8::MAX);
    let b = rng.random_range(0..=u8::MAX);
    let c = rng.random_range(0..=u8::MAX);

    let what = rng.random_range(0..super::OUTPUTS as u8); // which inner protein does this receptor bind to
    let how_mutch: u8 = if rand::random_bool(0.5) { 1 } else { 0 }; // does it increase or decrease the concentration of the inner protein
    let spec = rng.random_range(0..settings.possible_ligands()); // specificity of the receptor

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

    let spec = rng.random_range(0..settings.possible_ligands() as u16);
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
