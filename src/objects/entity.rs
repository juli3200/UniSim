use rand::Rng;
use ndarray::{Array1, Array2};
use super::{Entity, Ligand};
use crate::objects::Genome;
use crate::settings_::Settings;
use crate::world::{Border, Collision, Space};

const IDLE_COLLISION_TIMER: usize = 30; // number of updates to ignore collisions after a collision
const IDLE_BORDER_TIMER: usize = 10; // number of updates to ignore border collisions after a border collision


fn calculate_ligand_direction(entity: &Entity, position: &Array1<f32>) -> f64 {
    // calculate the angle between the entity's velocity and the direction to the ligand
    // return the angle in radians between 0 and PI
    
    let direction = (position - &entity.position).mapv(|x| x as f64);

    let v: Array1<f64> = entity.velocity.mapv(|x| x as f64);

    // dot product is enough because we only need unsigned angle
    let dot_product = direction.dot(&v);

    let direction_len = direction.mapv(|x| x.powi(2)).sum().sqrt();
    let v_len = v.mapv(|x| x.powi(2)).sum().sqrt();

    let angle = (dot_product / (direction_len * v_len)).acos();
    assert_ne!(angle, f64::NAN, "Angle should not be NaN");

    return angle;
}


impl Entity {
    pub(crate) fn new(id: usize, space: &mut Space, entities: &Vec<Entity>, settings: &Settings, genome: Option<Genome>) -> Result<Self, String> {
        let genome = match genome {
            Some(g) => g,
            None => Genome::random(settings),
        };


        let position = space
            .get_random_position(settings.spawn_size(), entities)?;


        // add the entity to the space
        space.add_entity(id, position.clone());

        // give a random velocity if settings.give_start_vel is true
        let velocity = if settings.give_start_vel() {
            let mut rng = rand::rng();
            let angle = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            vec![angle.cos(), angle.sin()].into()
        } else {
            Array1::zeros(2)
        };

        let mut e = Self {
            genome,

            id,
            energy: 0.0,

            age: 0,

            receptors: vec![0; settings.receptor_capacity()], // will be initialized later
            inner_protein_levels: [0; super::OUTPUTS],

            ligands_to_emit: vec![],

            position,
            size: settings.spawn_size(),
            velocity,
            acceleration: Array1::zeros(2),
            last_entity_collision: (0,0),
            last_border_collision: 0,
        };

        e.init_receptors(settings);
        
        Ok(e)

    }

    fn init_receptors(&mut self, settings: &Settings) {
        if settings.different_receptors() == 0 {
            self.receptors = vec![0; settings.receptor_capacity()];
            return; // no receptors to initialize
        }
        let mut rng = rand::rng();

        // this receptor array will be filled with receptors all over the membrane
        let mut receptors = Vec::with_capacity(settings.receptor_capacity());

        // this section receptors reference the *different* receptors in receptor_dna
        // extract receptor functions from receptor_dna
        let receptor_fns: Vec<Box<dyn Fn(f32) -> f64>> = self.genome.receptor_dna.iter().map(|&dna| super::receptor::extract_receptor_fns(dna)).collect();



        // e.g if receptor_capacity is 100 and there are 4 receptor types, each function is called 25 times
        // so every 4th receptor slot is reserved for the same receptor function
        // this ensures that the receptors are evenly distributed over the membrane

        for i in 0..(settings.receptor_capacity() / self.genome.receptor_dna.len()) {
            for r_type in 0..receptor_fns.len() {
                let p = receptor_fns[r_type]((i  * settings.different_receptors()) as f32); // probability to create a receptor here
                let create = rng.random_bool(p);

                if !create {
                    receptors.push(0); // no receptor
                    continue;
                }

                // create a receptor
                let receptor = u32::from_le_bytes(self.genome.receptor_dna[r_type].to_le_bytes()[4..8].try_into().unwrap());

                receptors.push(receptor);
            }
        }
        

    }

    pub(crate) fn update_physics(&mut self, space: &mut Space) {

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

    pub(crate) fn update_output(&mut self, settings: &Settings){
        // update the entity's biological state
        self.age += 1;
        
        // 0 MOVEMENT
        if self.inner_protein_levels[0] > self.genome.move_threshold {
            // run
            // TODOOOOOOOOOOOOOOOO how mutch acceleration?
            self.acceleration = &self.velocity * 0.1;

        } else if self.inner_protein_levels[0] <= self.genome.move_threshold {
            // tumble
            // TODOOO OVERHAUL TUMBLE
            // random rotation matrix
            let mut rng = rand::rng();
            
            if rng.random_bool(settings.tumble_chance()) {
                let angle: f32 = rng.random_range(-std::f32::consts::PI/ 4.0 ..std::f32::consts::PI / 4.0);

            let rotation_matrix: Array2<f32> = Array2::from_shape_vec((2, 2), vec![
                angle.cos(), -angle.sin(),
                angle.sin(), angle.cos(),
            ]).unwrap();

            self.velocity = rotation_matrix.dot(&self.velocity);
            }
            

        }

        // 1 EMIT LIGANDS

        if self.inner_protein_levels[1] > self.genome.ligand_emission_threshold {
            #[cfg(feature = "debug")]
            {
                println!("Entity {} emitted a ligand", self.id);
            }
        }
        // 2 REPRODUCTION



        // (FUTURE: KILLING OTHER ENTITIES)



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

        // TODOOOOOOOOO ligand function
    pub(crate) fn emit_ligands(&mut self) -> Vec<Ligand> {
        // Take the ligands from the entity and return them
        let ligands = self.ligands_to_emit.clone();
        self.ligands_to_emit.clear();
        ligands
    }

    pub(crate) fn receive_ligand(&mut self, ligand: &Ligand, settings: &Settings) -> bool{
        // return true if the ligand was processed, false if it was ignored (e.g. self-emitted ligand)

        // ignore self-emitted ligands (destroyed to prevent them from being stuck in the entity)
        if ligand.emitted_id == self.id {
            return true; 
        }

        // process the ligand message
        // for now, just increase energy based on message

        let angle = calculate_ligand_direction(self, &ligand.position);

        // handle the message

        let angle_index = (angle / std::f64::consts::PI * (settings.receptor_capacity() - 1) as f64).floor() as usize; // index in receptor array

        let receptor = self.receptors[angle_index];
        let bond_result = super::receptor::bond(receptor, ligand.message);

        if bond_result.is_none() {
            // bonding failed
            return false;
        }

        let (energy_change, concentration_change) = bond_result.unwrap();
        self.energy += energy_change;

        // change concentration
        let index = concentration_change.abs() as usize;
        let change: i16 = if concentration_change < 0 { -1 } else { 1 };

        assert!(index < self.inner_protein_levels.len(), "Concentration index out of bounds");

        // change concentration and clamp to range
        self.inner_protein_levels[index] = (self.inner_protein_levels[index] + change).clamp(settings.concentration_range().0, settings.concentration_range().1);
        return true;


    }
}