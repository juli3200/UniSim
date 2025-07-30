use super::{Entity};
use crate::world::Space;

impl Entity {
    pub fn new(id: usize, space: &mut Space, size: f32) -> Result<Self, String> {
        
        let position = space
            .get_random_position(size)?;

        Ok(Self { id, position, size })
    }
}