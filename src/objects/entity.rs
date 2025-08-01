use super::Entity;
use crate::world::{Border, Collision, Space};

impl Entity {
    pub fn new(id: usize, space: &mut Space, size: f32) -> Result<Self, String> {
        
        let position = space
            .get_random_position(size)?;

        let velocity = (0.0, 0.0); // initial velocity is set to zero
        Ok(Self { id, position, size, velocity })
    }

    fn resolve_collision(&mut self, collision: &Collision) {
        // only called when a collision is detected
        match collision {
            Collision::EntityCollision(velocity, mass) => {
                // resolve collision with other entity
                // elastic collision resolution

                // from https://en.wikipedia.org/wiki/Elastic_collision
                // TODO: implement proper collision resolution
            }


            Collision::BorderCollision(border) => {
                match border {
                    Border::Left => self.velocity.0 = -self.velocity.0, // reflect velocity
                    Border::Right => self.velocity.0 = -self.velocity.0,
                    Border::Top => self.velocity.1 = -self.velocity.1,
                    Border::Bottom => self.velocity.1 = -self.velocity.1,
                }
            }
            _ => {}
        }
    }
}