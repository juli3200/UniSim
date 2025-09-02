use rand::Rng;

use ndarray::Array1;
use super::Entity;
use crate::world::{Border, Collision, Settings, Space};

const IDLE_COLLISION_TIMER: usize = 10; // number of updates to ignore collisions after a collision

impl Entity {
    pub fn new(id: usize, space: &mut Space, entities: &Vec<Entity>, settings: &Settings) -> Result<Self, String> {

        let position = space
            .get_random_position(settings.spawn_size(), entities)?;


        // add the entity to the space
        space.add_entity(id, position.clone());

        // give a random velocity if settings.give_start_vel is true
        let velocity = if settings.give_start_vel() {
            let mut rng = rand::rng();
            Array1::from(vec![
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            ])
        } else {
            Array1::zeros(2)
        };

        Ok(Self {
            id,
            position,
            size: settings.spawn_size(),
            velocity,
            last_collision: None,
        })
    }

    pub(crate) fn update(&mut self, space: &mut Space, entities: &Vec<Entity>) {

        // update last_collision timer
        // timer is set by constant: IDLE_COLLISION_TIMER
        if let Some((last_id, last_time)) = self.last_collision {
            if last_time <= 1{ self.last_collision = None; }
            else { self.last_collision = Some((last_id, last_time -1)); }
        }

        let fps = 1.0 / space.settings.fps();

        // implement acceleration 
        // TODOOOO
        let old_position: Array1<f32> = self.position.clone();

        // update the entity's position based on its velocity
        self.position.scaled_add(fps * space.settings.velocity(), &self.velocity);

        // check for collisions with the space boundaries
        let collision = space.check_position(self.position.clone(), Some(self.size), Some(self.id), entities);

        space.update_entity_position(self.id, old_position, self.position.clone());

        // if no collision, return early
        if let Collision::NoCollision = collision {
            return;
        }

        self.resolve_collision(collision);

    }

    fn resolve_collision(&mut self, collision: Collision) {
        // only called when a collision is detected
        match collision {
            Collision::EntityCollision(other_velocity, mass, other_position, id) => {
                // check if the colliding entity is the last collided entity
                // this is used to avoid jittering
                if let Some((last_id, last_time)) = self.last_collision {
                    if last_id == id && last_time > 0 {
                        // skip update if the entity just collided with the same entity
                        println!("Skipping update for entity {}: collided with itself", id);
                        return;
                    }
                }

                // resolve collision with other entity
                // elastic collision resolution

                // source:
                // https://www.vobarian.com/collisions/2dcollisions2.pdf

                let v1 = self.velocity.clone();
                let v2 = other_velocity.clone();

                let m1 = self.size.powi(2); /* * std::f32::consts::PI; */// mass of this entity
                let m2 = mass; // mass of the other entity

                let delta_v = &v1 - &v2;
                let delta_p = &self.position - other_position;


                let new_v1: Array1<f32> = v1 - (2.0 * m2 / (m1 + m2)) *
                 (delta_v.dot(&delta_p) / delta_p.dot(&delta_p)) * delta_p;
                                        // delta_p.dot(&delta_p) is the squared norm 
                
                assert!(new_v1.len() == 2, "Velocity should be a 2D vector");
                self.velocity = new_v1;

                self.last_collision = Some((id, IDLE_COLLISION_TIMER));

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