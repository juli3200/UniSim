use crate::{objects::{Entity, Ligand}, prelude::Settings, world::World};

use std::time::{SystemTime, UNIX_EPOCH};

pub const ENTITY_BUF_SIZE: (usize, usize) = (20, 20);
pub const LIGAND_BUF_SIZE: (usize, usize) = (10, 22);
pub const WORLD_BUF_ADD: (usize, usize) = (17, 37);
pub const SETTINGS_BUF_SIZE: (usize, usize) = (0, 20);
pub const HEADER_SIZE: usize = 53;

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
    buffer.extend(&world.settings.gravity().iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()); // gravity 8 bytes
    buffer.extend(&world.settings.drag().to_le_bytes()); // drag 4 bytes
    // add other settings

    
    // 4 bytes
    buffer.extend(&(ENTITY_BUF_SIZE.0 as u8).to_le_bytes());
    buffer.extend(&(ENTITY_BUF_SIZE.1 as u8).to_le_bytes());
    buffer.extend(&(LIGAND_BUF_SIZE.0 as u8).to_le_bytes());
    buffer.extend(&(LIGAND_BUF_SIZE.1 as u8).to_le_bytes());

    Ok(buffer)
}

pub trait Save {
    // used to store the object position size ... to be displayed
    fn serialize(&self) -> Result<Vec<u8>, String>;
    // used store the whole object
    fn pause_serialize(&self) -> Result<Vec<u8>, String>;
}

impl<T: Save> Save for Vec<T> {
    fn serialize(&self) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        for item in self {
            buffer.extend(item.serialize()?);
        }
        Ok(buffer)
    }

    fn pause_serialize(&self) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        for item in self {
            buffer.extend(item.pause_serialize()?);
        }
        Ok(buffer)
    }
}

impl Save for Entity {
    fn serialize(&self) -> Result<Vec<u8>, String> {
        // collect all the data
        let buffer_vec = {
            // position 8 bytes
            self.position.iter().flat_map(|x| x.to_le_bytes()).chain(
            // size 4 bytes
                self.size.to_le_bytes().iter().cloned()
            ).chain(
            // velocity 8 bytes
                self.velocity.iter().flat_map(|x| x.to_le_bytes())
            ).collect::<Vec<u8>>()
        };

        if buffer_vec.len() != ENTITY_BUF_SIZE.0 {
            return Err("Invalid buffer length entity".to_string());
        }

        Ok(buffer_vec)
    }

    fn pause_serialize(&self) -> Result<Vec<u8>, String> {
        self.serialize()
    }

}

impl Save for Ligand {
    fn serialize(&self) -> Result<Vec<u8>, String> {
        let buffer_vec: Vec<u8> = self.position.iter().flat_map(|x| x.to_le_bytes()).chain(self.spec.to_le_bytes()).collect();


        if buffer_vec.len() != LIGAND_BUF_SIZE.0 {
            return Err("Invalid buffer length ligand".to_string());
        }

        Ok(buffer_vec)
    }

    fn pause_serialize(&self) -> Result<Vec<u8>, String> {
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
    fn serialize(&self) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        buffer.push(false as u8); // not a pause file 1 byte


        // time
        buffer.extend(&self.time.to_le_bytes()); // 4 bytes


        // entities
        buffer.extend(&(self.population_size as u32).to_le_bytes()); // 4 bytes
        buffer.extend(self.entities.serialize()?);

        let ligands_count = if cfg!(feature = "save_ligands") {
            self.ligands_count
        } else {
            0
        };

        // ligands are only saved in test mode for debugging purposes
        #[cfg(feature = "save_ligands")]
        {
        buffer.extend(&(self.ligands_count as u32).to_le_bytes()); // 4 bytes
        buffer.extend(self.ligands.serialize()?);
        }

        #[cfg(not(feature = "save_ligands"))]
        {
            buffer.extend(&0u32.to_le_bytes()); // 4 bytes for ligands count
        }

        // Insert the total buffer length at the beginning
        
        let total_len = (buffer.len() as u32).to_le_bytes();
        buffer.splice(0..0, total_len.iter().cloned()); 


        if buffer.len() != self.population_size * ENTITY_BUF_SIZE.0 + // entities
            ligands_count * LIGAND_BUF_SIZE.0 /* ligands */ + WORLD_BUF_ADD.0 {
                return Err("Invalid buffer length world".to_string());
        }


        Ok(buffer)
    }

    fn pause_serialize(&self) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        buffer.push(true as u8); // pause file 1 byte

        // add changeable settings
        buffer.extend(self.settings.pause_serialize()?); // 16 bytes

        // time, counter
        buffer.extend(&self.time.to_le_bytes()); // 4 bytes
        buffer.extend(&self.counter.to_le_bytes()); // 4 bytes


        // entities
        buffer.extend(&(self.population_size as u32).to_le_bytes()); // 4 bytes
        buffer.extend(self.entities.serialize()?);

        // ligands
        buffer.extend(&(self.ligands_count as u32).to_le_bytes()); // 4 bytes
        buffer.extend(self.ligands.serialize()?);

        let total_len = (buffer.len() as u32 + 4).to_le_bytes();
        buffer.splice(0..0, total_len.iter().cloned()); 


        if buffer.len() != self.population_size * ENTITY_BUF_SIZE.0 + // entities
            self.ligands_count * LIGAND_BUF_SIZE.0 /* ligands */ + WORLD_BUF_ADD.1 {
                return Err("Invalid buffer length world".to_string());
        }

        Ok(buffer)
    }
}


impl Save for Settings{
    fn serialize(&self) -> Result<Vec<u8>, String> {
        let buffer = Vec::new();


        Ok(buffer)
    }

    fn pause_serialize(&self) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        // fps 4 bytes
        buffer.extend(&self.fps().to_le_bytes());

        // velocity 4 bytes
        buffer.extend(&self.velocity().to_le_bytes());

        // gravity 4 bytes
        buffer.extend(&self.gravity().iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());

        // drag 4 bytes
        buffer.extend(&self.drag().to_le_bytes());

        if buffer.len() != SETTINGS_BUF_SIZE.0 {
            return Err("Invalid buffer length settings".to_string());
        }

        Ok(buffer)
    }
}
