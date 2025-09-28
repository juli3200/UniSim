use rand::Rng;

use ndarray::Array1;
use super::{Entity, Ligand};
use crate::settings_::Settings;
use crate::world::{Border, Collision, Space};

const IDLE_COLLISION_TIMER: usize = 30; // number of updates to ignore collisions after a collision
const IDLE_BORDER_TIMER: usize = 10; // number of updates to ignore border collisions after a border collision


fn calculate_ligand_direction(entity: &Entity, position: &Array1<f32>) -> f32 {
    assert_eq!(position.len(), 2, "Position should be a 2D vector");
    assert_eq!(entity.position.len(), 2, "Entity position should be a 2D vector");

    // not neccessary to normalize, because 

    let direction = position - &entity.position;

    let v: Array1<f32> = entity.velocity.clone();

    // cross product
    let cross_product = direction[0] * v[1] - direction[1] * v[0];

    // dot product
    let dot_product = direction[0] * v[0] + direction[1] * v[1];

    let angle = cross_product.atan2(dot_product);
    assert_ne!(angle, f32::NAN, "Angle should not be NaN");

    return angle;
}


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
            energy: 0.0,
            dna: vec![],
            age: 0,
            reproduction_rate: 0.0,

            position,
            size: settings.spawn_size(),
            velocity,
            acceleration: Array1::zeros(2),
            last_entity_collision: (0,0),
            last_border_collision: 0,
        })
    }

    pub(crate) fn update(&mut self, space: &mut Space) {

        // update last_collision timer
        // timer is set by constant: IDLE_COLLISION_TIMER
        if self.last_entity_collision.1 > 0 {
            self.last_entity_collision.1 -= 1;
        }

        if self.last_border_collision > 0 {
            self.last_border_collision -= 1;
        }


        let fps = 1.0 / space.settings.fps();

        
        // acceleration
        self.velocity.scaled_add(fps, &self.acceleration);
        // friction
        self.velocity *= 1.0 - space.settings.friction() * fps;
        // gravity
        self.velocity.scaled_add(fps, &space.settings.gravity());

        // update the entity's position based on its velocity
        self.position.scaled_add(fps * space.settings.velocity(), &self.velocity);

        let old_position: Array1<f32> = self.position.clone();

        space.update_entity_position(self.id, old_position, self.position.clone());

    }

    // TODOOOOOOOOO ligand function
    pub(crate) fn emit_ligands(&mut self) -> Vec<Ligand> {
        // Take the ligands from the entity and return them
        return vec![];

        todo!()
    }

    pub(crate) fn receive_ligand(&mut self, message: u32, position: Array1<f32>) -> Result<(), String> {
        // process the ligand message
        // for now, just increase energy based on message

        let angle = calculate_ligand_direction(self, &position);

        #[cfg(feature = "debug")]
        println!("Entity {} received ligand with message {} from angle {}", self.id, message, angle);

        

        todo!()

    }

    pub(crate)fn resolve_collision(&mut self, space: &mut Space, entities: &Vec<Entity>) {

        // check for collisions with the space boundaries
        let collision = space.check_position(self.position.clone(), Some(self.size), Some(self.id), entities);

        
        match collision {
            Collision::EntityCollision(other_velocity, mass, other_position, id) => {
                // check if the colliding entity is the last collided entity
                // this is used to avoid jittering
                if self.last_entity_collision.0 == id && self.last_entity_collision.1 > 0 {
                    // skip update if the entity just collided with the same entity
                    return;
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
                if self.velocity[0].is_nan() || self.velocity[1].is_nan() {
                    panic!("NaN velocity detected after collision resolution");
                }

                self.last_entity_collision = (id, IDLE_COLLISION_TIMER);

            }


            Collision::BorderCollision(border) => {
                // check if the entity just collided with a border
                if self.last_border_collision > 0 {
                    return;
                }
                
                match border {
                    Border::Left => self.velocity[0] = -self.velocity[0], // reflect velocity
                    Border::Right => self.velocity[0] = -self.velocity[0],
                    Border::Top => self.velocity[1] = -self.velocity[1],
                    Border::Bottom => self.velocity[1] = -self.velocity[1],
                }

                self.last_border_collision = IDLE_BORDER_TIMER;
            }
            _ => {}
        }
    }
}