use ndarray::Array1;
use rand::Rng;
use crate::{objects::Entity, world::{Border, Collision, Space}};

use super::{Ligand, LigandSource};

impl LigandSource {
    pub fn new(position: Array1<f32>, emission_rate: f32, ligand_spec: u16, ligand_energy: f32) -> Self {
        Self {
            position,
            emission_rate,
            ligand_energy,
            ligand_spec
        }
    }

    pub(crate) fn emit_ligands(&self, dt: f32) -> Vec<Ligand> {
        let quantity;
        let mut rng = rand::rng();


        // emit 'quantity' ligands per frame
        if self.emission_rate * dt < 1.0 {
            if !rng.random_bool((self.emission_rate * dt) as f64) {
                return Vec::new();
            } else {
                quantity = 1;
            }
        } else {
            quantity = (self.emission_rate * dt).floor() as usize;
        }

        let mut ligands = Vec::with_capacity(quantity);



        for _ in 0..quantity {

            let angle = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            let velocity: Array1<f32> = vec![angle.cos(), angle.sin()].into();
            

            ligands.push(Ligand::new(
                0, // no entity emitted this ligand
                self.ligand_energy,
                self.ligand_spec,
                self.position.clone(),
                velocity
            ));
        }

        ligands

    }
}


impl Ligand {

    pub fn new(emitted_id: usize, energy: f32, spec:u16, position: Array1<f32>, velocity: Array1<f32>) -> Self {
        Self {
            emitted_id,
            spec,
            energy,
            position,
            velocity,
        }
    }

    pub(crate) fn update(&mut self, space: &Space, entities: &Vec<Entity>, dt: f32) -> Option<usize> {
        // update position based on velocity and dt
        // position = position + velocity * dt * space.settings.velocity()
        self.position.scaled_add(dt* space.settings.ligand_velocity(), &self.velocity);

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

    pub(crate ) fn re_emit(&mut self, entity: & Entity, dt: f32) {

        let dx_normal = (self.position[0] - entity.position[0])/ entity.size;
        let dy_normal = (self.position[1] - entity.position[1])/ entity.size;

        let dot = self.velocity[0] * dx_normal + self.velocity[1] * dy_normal;
        self.velocity[0] = self.velocity[0] - 2.0 * dot * dx_normal;
        self.velocity[1] = self.velocity[1] - 2.0 * dot * dy_normal;

        // move the ligand two steps away from the entity to avoid immediate re-collision
        self.position.scaled_add(2.0 * dt, &self.velocity);


    }

}
