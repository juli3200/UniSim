use ndarray::Array1;
use rand::Rng;
use crate::{objects::Entity, world::{Border, Collision, Space}};

use super::{Ligand, LigandSource};

impl LigandSource {
    pub fn new(position: Array1<f32>, emission_rate: f32, ligand_message: u32) -> Self {
        Self {
            position,
            emission_rate,
            ligand_message,
        }
    }

    pub(crate) fn emit_ligands(&self, fps: f32) -> Vec<Ligand> {
        let quantity = (self.emission_rate / fps) as usize; 
        // emit 'quantity' ligands per frame

        let mut ligands = Vec::with_capacity(quantity);

        let mut rng = rand::rng();

        for _ in 0..quantity {

            let angle = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            let velocity: Array1<f32> = vec![angle.cos(), angle.sin()].into();
            

            ligands.push(Ligand::new(
                usize::MAX, // no entity emitted this ligand
                self.ligand_message,
                self.position.clone(),
                velocity
            ));
        }

        ligands

    }
}

impl Ligand {

    pub fn new(emitted_id: usize, message: u32, position: Array1<f32>, velocity: Array1<f32>) -> Self {
        Self {
            emitted_id,
            message,
            position,
            velocity,
        }
    }

    pub(crate) fn update(&mut self, space: &Space, entities: &Vec<Entity>, dt: f32) -> Option<usize> {
        // update position based on velocity and dt
        // position = position + velocity * dt * space.settings.velocity()
        self.position.scaled_add(dt* space.settings.velocity(), &self.velocity);

        // check for collisions
        let collision = space.check_position(self.position.clone(), None, None, entities);

        let mut collided_entity_id: Option<usize> = None;

        match collision {
            Collision::BorderCollision(border) => {
                match border {
                    Border::Left | Border::Right => {
                        self.velocity[0] = -self.velocity[0];
                    }
                    Border::Top | Border::Bottom => {
                        self.velocity[1] = -self.velocity[1];
                    }
                }
            }
            Collision::EntityCollision(_, _, _, entity_id) => {
                // return the id to be consumed by the entity
                collided_entity_id = Some(entity_id);
            }
            Collision::NoCollision => {}
        }

        collided_entity_id

    }

}