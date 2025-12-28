#![allow(unused_mut)]

use crate::{objects::{Entity, Ligand}, prelude::Settings, world::World};

use std::time::{SystemTime, UNIX_EPOCH};

pub const ENTITY_BUF_SIZE: (usize, usize) = (32 + super::objects::OUTPUTS * 2, 0);
pub const LIGAND_BUF_SIZE: (usize, usize) = (10, 22);
pub const WORLD_BUF_ADD: (usize, usize) = (17, 37);
pub const SETTINGS_BUF_SIZE: (usize, usize) = (0, 20);
pub const HEADER_SIZE: u8 = 126; // to keep it backwards compatible

pub(crate) fn serialize_header(world: &World) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();

    let time: u64 = SystemTime::now().duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    buffer.push(HEADER_SIZE as u8); // header size 1 byte
    buffer.extend(&[0u8; 4]); // jumper to the latest save
    buffer.extend(&time.to_le_bytes()); // time 8 bytes

    buffer.extend(&world.settings.dimensions().0.to_le_bytes()); // width 4 bytes
    buffer.extend(&world.settings.dimensions().1.to_le_bytes()); // height 4 bytes
    buffer.extend(&world.settings.spawn_size().to_le_bytes()); // spawn size 4 bytes
    buffer.extend(&(world.settings.store_capacity() as u32).to_le_bytes()); // store capacity 4 bytes
    buffer.extend(&world.settings.fps().to_le_bytes()); // fps 4 bytes
    buffer.extend(&world.settings.velocity().to_le_bytes()); // velocity 4 bytes
    buffer.extend(&world.settings.general_force().0.to_le_bytes());
    buffer.extend(&world.settings.general_force().1.to_le_bytes()); // gravity 8 bytes
    buffer.extend(&world.settings.drag().to_le_bytes()); // drag 4 bytes

    buffer.extend(vec![0u8; 32]); // reserved 32 bytes

    // entity bio settings
    buffer.extend(&world.settings.ligands_per_entity().to_le_bytes()); // ligand variety 4 bytes per entity
    buffer.extend(&world.settings.receptor_types_per_entity().to_le_bytes()); // receptors per entity 4 bytes per entity
    buffer.extend(vec![0u8; 32]); // reserved 32 bytes

    // add other settings
    buffer.push(ENTITY_BUF_SIZE.0 as u8);
    buffer.push(ENTITY_BUF_SIZE.1 as u8);
    buffer.push(LIGAND_BUF_SIZE.0 as u8);
    buffer.push(LIGAND_BUF_SIZE.1 as u8);

    buffer.push(super::objects::OUTPUTS as u8); // number of inner proteins 1 byte
    

    if buffer.len() != HEADER_SIZE as usize {
        return Err("Wrong Header Size".to_string());
    }

    Ok(buffer)
}

pub trait Save {
    // used to store the object position size ... to be displayed
    fn serialize(&self, save_genome: bool) -> Result<Vec<u8>, String>;
    // used store the whole object
    fn pause_serialize(&mut self, save_genome: bool) -> Result<Vec<u8>, String>;
}

impl<T: Save> Save for Vec<T> {
    fn serialize(&self, save_genome: bool) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        for item in self {
            buffer.extend(item.serialize(save_genome)?);
        }
        Ok(buffer)
    }

    fn pause_serialize(&mut self, save_genome: bool) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        for item in self {
            buffer.extend(item.pause_serialize(save_genome)?);
        }
        Ok(buffer)
    }
}

impl Save for Entity {
    fn serialize(&self, save_genome: bool) -> Result<Vec<u8>, String> {
        let mut buffer_vec = vec![];
        buffer_vec.extend(self.position.iter().flat_map(|x| x.to_le_bytes())); // position 8 bytes
        buffer_vec.extend(self.velocity.iter().flat_map(|x| x.to_le_bytes())); // velocity 8 bytes
        buffer_vec.extend(self.size.to_le_bytes()); // size 4 bytes
        buffer_vec.extend(self.energy.to_le_bytes()); // energy 4 bytes

        buffer_vec.extend((self.id as u32).to_le_bytes()); // id 4 bytes

        for level in self.inner_protein_levels.iter() {
            buffer_vec.extend(level.to_le_bytes()); // concentration levels 2 bytes each
        }

        buffer_vec.extend((self.received_ligands.len() as u32).to_le_bytes()); // number of received ligands 4 bytes
        buffer_vec.extend(&self.received_ligands);

        let mut genome_bytes = vec![];

        if save_genome {
            genome_bytes.extend(self.genome.serialize(save_genome)?);
            buffer_vec.extend(genome_bytes.clone());
        }

        if buffer_vec.len() != ENTITY_BUF_SIZE.0  + self.received_ligands.len()  + genome_bytes.len() {
            return Err("Invalid buffer length entity".to_string());
        }

        Ok(buffer_vec)
    }

    fn pause_serialize(&mut self, save_genome: bool) -> Result<Vec<u8>, String> {
        self.serialize(save_genome)
    }

}

impl Save for Ligand {
    fn serialize(&self, _save_genome: bool) -> Result<Vec<u8>, String> {
        let buffer_vec: Vec<u8> = self.position.iter().flat_map(|x| x.to_le_bytes()).chain(self.spec.to_le_bytes()).collect();


        if buffer_vec.len() != LIGAND_BUF_SIZE.0 {
            return Err("Invalid buffer length ligand".to_string());
        }

        Ok(buffer_vec)
    }

    fn pause_serialize(&mut self, _save_genome: bool) -> Result<Vec<u8>, String> {
        // collect all the data
        let buffer_vec = {
            // position 8 bytes
            self.position.iter().flat_map(|x| x.to_le_bytes()).chain(
            // velocity 8 bytes
            self.velocity.iter().flat_map(|x| x.to_le_bytes()).chain(
            self.energy.to_le_bytes()).chain(
            self.spec.to_le_bytes())
            ).collect::<Vec<u8>>()
        };

        if buffer_vec.len() != LIGAND_BUF_SIZE.1 {
            return Err("Invalid buffer length ligand".to_string());
        }

        Ok(buffer_vec)
    }
}


impl Save for World {
    fn serialize(&self, save_genome: bool) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        //            2     1      0
        // info byte ... |save |genome| ->  ... reserved for future use
        let info_byte: u8 = (false as u8) << 1 | (self.save_genome as u8) << 0;
        buffer.push(info_byte);

        // time
        buffer.extend(&self.time.to_le_bytes()); // 4 bytes


        // entities
        buffer.extend(&(self.entities.len() as u32).to_le_bytes()); // 4 bytes
        let serial_entities = self.entities.serialize(save_genome)?;
        buffer.extend(&serial_entities);

        let ligands_count = if cfg!(feature = "save_ligands") {
            self.ligands.len()
        } else {
            0
        };

        // ligands are only saved in test mode for debugging purposes
        #[cfg(feature = "save_ligands")]
        {
        buffer.extend(&(ligands_count as u32).to_le_bytes()); // 4 bytes
        buffer.extend(self.ligands.serialize(save_genome)?);
        }

        #[cfg(not(feature = "save_ligands"))]
        {
            buffer.extend(&0u32.to_le_bytes()); // 4 bytes for ligands count
        }

        // Insert the total buffer length at the beginning
        
        let total_len = ((buffer.len() + 4) as u32).to_le_bytes();
        buffer.splice(0..0, total_len.iter().cloned()); 


        if buffer.len() != serial_entities.len() + // entities
            ligands_count * LIGAND_BUF_SIZE.0 /* ligands */ + WORLD_BUF_ADD.0 {
                return Err("Invalid buffer length world".to_string());
        }


        Ok(buffer)
    }

    fn pause_serialize(&mut self, save_genome: bool) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        //            2     1      0
        // info byte ... |save |genome| ->  ... reserved for future use 
        let info_byte: u8 = (true as u8) << 1 | (self.save_genome as u8) << 0;
        buffer.push(info_byte);

        // add changeable settings
        buffer.extend(self.settings.pause_serialize(save_genome)?); // 16 bytes

        // time, counter
        buffer.extend(&self.time.to_le_bytes()); // 4 bytes
        buffer.extend(&self.counter.to_le_bytes()); // 4 bytes


        // entities
        buffer.extend(&(self.population_size() as u32).to_le_bytes()); // 4 bytes
        buffer.extend(self.entities.serialize(save_genome)?);

        // ligands
        #[cfg(feature = "cuda")]
        {
            self.copy_ligands();
        }
        buffer.extend(&(self.ligands.len() as u32).to_le_bytes()); // 4 bytes
        buffer.extend(self.ligands.serialize(save_genome)?);

        let total_len = (buffer.len() as u32 + 4).to_le_bytes();
        buffer.splice(0..0, total_len.iter().cloned()); 


        if buffer.len() != self.population_size() * ENTITY_BUF_SIZE.0 + // entities
            self.ligands.len() * LIGAND_BUF_SIZE.0 /* ligands */ + WORLD_BUF_ADD.1 {
                return Err("Invalid buffer length world".to_string());
        }

        Ok(buffer)
    }
}


impl Save for Settings{
    fn serialize(&self, _save_genome: bool) -> Result<Vec<u8>, String> {
        let buffer = Vec::new();


        Ok(buffer)
    }

    fn pause_serialize(&mut self, _save_genome: bool) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        // fps 4 bytes
        buffer.extend(&self.fps().to_le_bytes());

        // velocity 4 bytes
        buffer.extend(&self.velocity().to_le_bytes());

        // gravity 4 bytes
        buffer.extend(&self.general_force().0.to_le_bytes());
        buffer.extend(&self.general_force().1.to_le_bytes());

        // drag 4 bytes
        buffer.extend(&self.drag().to_le_bytes());

        if buffer.len() != SETTINGS_BUF_SIZE.0 {
            return Err("Invalid buffer length settings".to_string());
        }

        Ok(buffer)
    }
}


impl Save for crate::objects::Genome{
    fn serialize(&self, _save_genome: bool) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        buffer.extend(&self.move_threshold.to_le_bytes()); // move threshold 2 bytes
        buffer.extend(&self.ligand_emission_threshold.to_le_bytes()); // ligand emission threshold 2 bytes

        for ligand in self.ligands.iter() {
            buffer.extend(&ligand.to_le_bytes()); // ligands 2 bytes each
        }

        for receptor in self.receptor_dna.iter() {
            buffer.extend(&receptor.to_le_bytes()); // receptor DNA 8 bytes each
        }
        Ok(buffer)
    }

    fn pause_serialize(&mut self, _save_genome: bool) -> Result<Vec<u8>, String> {
        self.serialize(_save_genome)
    }
}
