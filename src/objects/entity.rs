use ndarray::Array1;

use super::Entity;
use crate::world::{Border, Collision, Space};

impl Entity {
    pub fn new(id: usize, space: &mut Space, size: f32) -> Result<Self, String> {
        
        let position = space
            .get_random_position(size)?;

        let velocity = Array1::zeros(2); // initial velocity is set to zero
        Ok(Self { id, position, size, velocity })
    }

    fn resolve_collision(&mut self, collision: &Collision) {
        // only called when a collision is detected
        match collision {
            Collision::EntityCollision(other_velocity, mass, other_position) => {
                // resolve collision with other entity
                // elastic collision resolution

                // source:
                // https://www.vobarian.com/collisions/2dcollisions2.pdf

                let v1 = self.velocity.clone();
                let v2 = other_velocity.clone();

                let m1 = self.size.powi(2) * std::f32::consts::PI; // mass of this entity
                let m2 = *mass; // mass of the other entity

                let delta_v = &v1 - &v2;
                let delta_p = &self.position - other_position;


                let new_v1: Array1<f32> = v1 - (2.0 * m2 / (m1 + m2)) *
                 (delta_v.dot(&delta_p) / delta_p.dot(&delta_p)) * delta_p;
                                        // delta_p.dot(&delta_p) is the squared norm 
                
                assert!(new_v1.len() == 2, "Velocity should be a 2D vector");
                self.velocity = new_v1;

            }


            Collision::BorderCollision(border) => {
                match border {
                    Border::Left => self.velocity[0] = -self.velocity[0], // reflect velocity
                    Border::Right => self.velocity[0] = -self.velocity[0],
                    Border::Top => self.velocity[1] = -self.velocity[1],
                    Border::Bottom => self.velocity[1] = -self.velocity[1],
                }
            }
            _ => {}
        }
    }
}