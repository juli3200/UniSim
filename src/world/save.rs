use crate::objects::{Entity, Ligand};

const ENTITY_BUF_SIZE: (usize, usize) = (20, 20);
const LIGAND_BUF_SIZE: (usize, usize) = (8, 16);

pub trait Save {
    // used to store the entity position size ... to be displayed
    fn serialize(&self) -> Result<Vec<u8>, String>;
    // used store the whole entity 
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
            return Err("Invalid buffer length".to_string());
        }

        Ok(buffer_vec)
    }

    fn pause_serialize(&self) -> Result<Vec<u8>, String> {
        self.serialize()
    }

}

impl Save for Ligand {
    fn serialize(&self) -> Result<Vec<u8>, String> {
        let buffer_vec: Vec<u8> = self.position.iter().flat_map(|x| x.to_le_bytes()).collect();

        if buffer_vec.len() != LIGAND_BUF_SIZE.0 {
            return Err("Invalid buffer length".to_string());
        }

        Ok(buffer_vec)
    }

    fn pause_serialize(&self) -> Result<Vec<u8>, String> {
        // collect all the data
        let buffer_vec = {
            // position 8 bytes
            self.position.iter().flat_map(|x| x.to_le_bytes()).chain(
            // velocity 8 bytes
                self.velocity.iter().flat_map(|x| x.to_le_bytes())
            ).collect::<Vec<u8>>()
        };

        if buffer_vec.len() != LIGAND_BUF_SIZE.1 {
            return Err("Invalid buffer length".to_string());
        }

        Ok(buffer_vec)
    }
}