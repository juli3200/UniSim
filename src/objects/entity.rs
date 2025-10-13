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

    let angle = (dot_product / (direction_len * entity.speed as f64)).acos();
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
            speed: 0.0,
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

        self.receptors = receptors;
        

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


        let dt = 1.0 / space.settings.fps();

        
        // acceleration
        self.velocity.scaled_add(dt, &self.acceleration);

        // update current speed
        self.speed = self.velocity.mapv(|x| x.powi(2)).sum().sqrt();

        // drag

        // acceleration due to drag would be: 1/2 * cd * p * A * v^2 / m => 1/2 * cd * p * A * v^2 / (pi * size^2)
        // A = 2 * size (diameter in 2D)
        // so size cancels out one size in the denominator and the 2  cancels the 1/2
        // so we get cd * p * v^2 / (pi * size)
        // since p = 1, we get 
        // cd * v^2 / (pi * size)

        let cd = space.settings.drag(); // drag coefficient        

        let ad: Array1<f32> = &self.velocity * (cd * self.speed / (std::f32::consts::PI * self.size)) * dt; // acceleration due to drag

        self.velocity -= &ad;

        // gravity
        self.velocity.scaled_add(dt, &Array1::from_vec(space.settings.gravity()));

        // update the entity's position based on its velocity
        self.position.scaled_add(dt * space.settings.velocity(), &self.velocity);

        let old_position: Array1<f32> = self.position.clone();

        space.update_entity_position(self.id, old_position, self.position.clone());

    }

    pub(crate) fn update_output(&mut self, settings: &Settings){

        let mut rng = rand::rng();

        // update the entity's biological state
        self.age += 1;
        
        // 0 MOVEMENT
        if self.inner_protein_levels[0] > self.genome.move_threshold {
            // run
            // TODOOOOOOOOOOOOOOOO how mutch acceleration?
            self.acceleration = &self.velocity * 0.1 / (self.size * 2.0); // acceleration proportional to velocity and inversely proportional to size

        } else if self.inner_protein_levels[0] <= self.genome.move_threshold {
            // tumble
            // TODOOO OVERHAUL TUMBLE
            // random rotation matrix
            
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
            // determine what ligand to emit
            let step: i16 = ((settings.concentration_range().1 - self.genome.ligand_emission_threshold) as f32 / settings.ligand_types() as f32).floor() as i16;
            let l_index = ((self.inner_protein_levels[1] - self.genome.ligand_emission_threshold) / step) as usize;
            let (energy, spec) = if l_index >= settings.ligand_types() as usize {
                self.genome.ligands[settings.ligand_types() as usize - 1]
            } else {
                self.genome.ligands[l_index]
            };
            let random_angle: f32 = rng.random_range(-std::f32::consts::PI ..std::f32::consts::PI);
            let direction: Array1<f32> = vec![random_angle.cos(), random_angle.sin()].into();
            let position: Array1<f32> = &self.position + &direction * self.size; // emit from the edge of the entity
            let velocity: Array1<f32> = &self.velocity + &direction ;

            self.ligands_to_emit.push(Ligand::new(self.id, energy, spec, position, velocity));

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
        let energy_cost: f32 = ligands.iter().map(|x| x.energy).sum();

        self.energy -= energy_cost;

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
        let bond_result = super::receptor::bond(receptor, ligand.spec);

        if bond_result.is_none() {
            // bonding failed
            return false;
        }

        let concentration_change = bond_result.unwrap();
        self.energy += ligand.energy;

        // change concentration
        let index = concentration_change.abs() as usize;
        let change: i16 = if concentration_change < 0 { -1 } else { 1 };

        assert!(index < self.inner_protein_levels.len(), "Concentration index out of bounds");

        // change concentration and clamp to range
        self.inner_protein_levels[index] = (self.inner_protein_levels[index] + change).clamp(settings.concentration_range().0, settings.concentration_range().1);
        return true;


    }

    pub(crate) fn print_stats(&self) {
        println!("Entity ID: {}", self.id);
        println!("Age: {}", self.age);
        println!("Energy: {:.2}", self.energy);
        println!("Position: ({:.2}, {:.2})", self.position[0], self.position[1]);
        println!("Velocity: ({:.2}, {:.2})", self.velocity[0], self.velocity[1]);
        println!("Speed: {:.2}", self.speed);
        println!("Size: {:.2}", self.size);
        println!("Receptors: {:?}", self.receptors);
        println!("Receptor DNA: {:?}", self.genome.receptor_dna);
        println!("Inner Protein Levels: {:?}", self.inner_protein_levels);
        println!("Genome Ligands: {:?}", self.genome.ligands);
        println!("Move Threshold: {}", self.genome.move_threshold);
        println!("Ligand Emission Threshold: {}", self.genome.ligand_emission_threshold);
        println!("Reproduction Threshold: {}", self.genome.reproduction_threshold);
    }
}