use rand::Rng;
use ndarray::{Array1, Array2};
use super::{Entity, Ligand};
use crate::objects::{Genome, PlasmidPackage};
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
    if angle.is_nan() {
        eprintln!("NaN angle detected in ligand direction calculation");
        return 0.0;
    }
    


    return angle;
}


impl Entity {
    pub(crate) fn new(id: usize, space: &mut Space, entities: &Vec<Entity>, settings: &Settings) -> Result<Self, String> {
        let genome = Genome::random(settings);
        let position: Array1<f32> = space.get_random_position(settings.spawn_size(), entities)?;

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
            energy: settings.spawn_size().powi(2) * std::f32::consts::PI, // initial energy set to area of the entity

            age: 0,

            receptors: vec![u32::MAX; settings.receptors_per_entity()], // will be initialized later
            inner_protein_levels: [0; super::OUTPUTS],

            ligands_to_emit: vec![],
            received_ligands: vec![],

            plasmid_bridge: false,

            position,
            size: settings.spawn_size(),
            velocity,
            speed: 0.0,
            acceleration: Array1::zeros(2),
            last_entity_collision: (0,0),
            last_border_collision: 0,

            #[cfg(feature = "cuda")]
            cuda_receptor_index: None,
        };

        e.init_receptors(settings);
        
        Ok(e)

    }

    fn init_receptors(&mut self, settings: &Settings) {
        if settings.receptor_types_per_entity() == 0 {
            self.receptors = vec![u32::MAX; settings.receptors_per_entity()];
            return; // no receptors to initialize
        }
        let mut rng = rand::rng();

        // this receptor array will be filled with receptors all over the membrane
        let mut receptors = Vec::with_capacity(settings.receptors_per_entity());

        // this section receptors reference the *different* receptors in receptor_dna
        // extract receptor functions from receptor_dna
        let receptor_fns: Vec<Box<dyn Fn(f32) -> f64>> = self.genome.receptor_dna.iter().map(|&dna| super::receptor::extract_receptor_fns(dna)).collect();



        // e.g if receptor_capacity is 100 and there are 4 receptor types, each function is called 25 times
        // so every 4th receptor slot is reserved for the same receptor function
        // this ensures that the receptors are evenly distributed over the membrane

        for i in 0..(settings.receptors_per_entity() / self.genome.receptor_dna.len()) {
            for r_type in 0..receptor_fns.len() {
                let p = receptor_fns[r_type]((i  * settings.receptor_types_per_entity() as usize) as f32); // probability to create a receptor here
                let create = rng.random_bool(p);

                if !create {
                    receptors.push(u32::MAX); // no receptor
                    continue;
                }

                // create a receptor
                let receptor = u32::from_le_bytes(self.genome.receptor_dna[r_type].to_le_bytes()[4..8].try_into().unwrap());

                receptors.push(receptor);
            }
        }

        self.receptors = receptors;
        

    }

    pub(crate) fn reproduce(&mut self, id: usize, space: &mut Space, entities: &Vec<Entity>, settings: &Settings) -> Option<Self> {
        if self.size < settings.max_size() {
            return None; // entity is not big enough to reproduce
        }

        let mut rng = rand::rng();

        let energy = self.energy / 2.0; // offspring gets half of parent's energy
        let size = energy.sqrt() / std::f32::consts::PI; // offspring has the default size for its energy

        let mut c = 0;
        let increase_search_interval = 1000;
        let mut size_multiplier = 1.0;

        let (position, velocity) = loop {
            c += 1;
            if c > increase_search_interval {
                size_multiplier += 0.5; // increase search area
                c = 0;
            }
            let angle: f32 = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
            let direction: Array1<f32> = vec![angle.cos(), angle.sin()].into();
            let position: Array1<f32> = &self.position + &direction * self.size * 2.0 * size_multiplier; // spawn at a distance of 2x size from parent
            let velocity: Array1<f32> = self.speed * &direction;

            if let Collision::NoCollision = space.check_position(position.clone(), Some(size), None, entities) {
                break (position, velocity);
            }
        };

        self.energy = energy; // parent keeps half of its energy
        // self.age = 0; // reset age to prevent immediate death after reproduction ?? todo: is this desired?
        self.size = size;

        // add the entity to the space
        space.add_entity(id, position.clone());

        let new_genome = self.genome.mutate(settings);

        let mut e = Self {
            id: id,
            genome: new_genome,
            energy,
            age: 0,
            receptors: Vec::with_capacity(settings.receptors_per_entity()),
            inner_protein_levels: [0; super::OUTPUTS],
            ligands_to_emit: vec![],
            received_ligands: vec![],
            plasmid_bridge: false,
            position,
            size,
            velocity,
            speed: self.speed,
            acceleration: Array1::zeros(2),
            last_entity_collision: (0,0),
            last_border_collision: 0,
            #[cfg(feature = "cuda")]
            cuda_receptor_index: None,
        };

        e.init_receptors(settings);

        Some(e)

    }

    pub(crate) fn update_physics(&mut self, space: &Space) -> Array1<f32> {

        // clear received ligands
        self.received_ligands.clear();

        // update last_collision timer
        // timer is set by constant: IDLE_COLLISION_TIMER
        if self.last_entity_collision.1 > 0 {
            self.last_entity_collision.1 -= 1;
        }

        if self.last_border_collision > 0 {
            self.last_border_collision -= 1;
        }

        let dt = 1.0 / space.settings.fps();

        // idle energy cost
        // energy cost proportional to area
        // todo: make this more biologically accurate
        self.energy -= space.settings.idle_energy_cost() * self.size.powi(2) * dt;
        self.size = (self.energy).sqrt()/ std::f32::consts::PI; // update size based on energy

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
        self.velocity.scaled_add(dt, &Array1::from_vec(vec![space.settings.general_force().0, space.settings.general_force().1]));

        // update the entity's position based on its velocity
        self.position.scaled_add(dt * space.settings.velocity(), &self.velocity);

        let old_position: Array1<f32> = self.position.clone();

        old_position

    }

    pub(crate) fn update_output(&mut self, settings: &Settings){

        let mut rng = rand::rng();

        // update the entity's biological state
        self.age += 1;
        
        // 0 MOVEMENT
        if self.inner_protein_levels[0] > self.genome.move_threshold {
            // run
            // TODOO how mutch acceleration? and energy cost?

            // acc = direction of velocity normalized * move_speed / energy
            self.acceleration = (&self.velocity / self.speed) * settings.entity_acceleration() / self.energy; // accelerate in the direction of movement
            self.energy -= settings.entity_run_energy_cost();

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

            
            self.energy -= settings.entity_tumble_energy_cost();

        }

        // 1 EMIT LIGANDS
        // check if inner protein level 1 is above ligand emission threshold
        let emit = self.inner_protein_levels[1] > self.genome.ligand_emission_threshold;

        // 2
        if emit {
            // determine what ligand to emit
            // step is the range in which the same ligand is emitted (e.g. if concentration range is 0-100 and there are 5 ligand types, step is 20)
            let step: i16 = (((settings.concentration_range().1 - settings.concentration_range().0) as f32 / settings.ligands_per_entity() as f32).floor() as i16).max(1);
            // find which ligand to emit based on concentration
            let l_index = ((self.inner_protein_levels[1] - self.genome.ligand_emission_threshold) / step) as usize;

            // get energy and spec of the ligand to emit
            let spec = if l_index >= settings.ligands_per_entity() as usize {
                self.genome.ligands[settings.ligands_per_entity() as usize - 1]
            } else {
                self.genome.ligands[l_index]
            };
            

            let random_angle: f32 = rng.random_range(-std::f32::consts::PI ..std::f32::consts::PI);
            let direction: Array1<f32> = vec![random_angle.cos(), random_angle.sin()].into();
            let position: Array1<f32> = &self.position + &direction * self.size; // emit from the edge of the entity
            let velocity: Array1<f32> = &self.velocity + &direction ;

            self.ligands_to_emit.push(Ligand::new(self.id, spec, position, velocity, settings));
        }

        // 3. PLASMID BRIDGE
        if self.inner_protein_levels[3] > self.genome.plasmid_threshold {
            self.plasmid_bridge = true;
        } else {
            self.plasmid_bridge = false;
        }


    }

    pub(crate) fn receive_plasmid(&mut self, plasmid: u16, settings: &Settings) {
        // check if plasmid is already present
        if self.genome.plasmids.contains(&plasmid) {
            return; // plasmid already present
        }

        // check if there is space for the plasmid
        if self.genome.plasmids.len() == settings.max_plasmid_count() as usize {
            self.genome.plasmids.remove(0); // remove oldest plasmid
        }

        // add the plasmid
        self.genome.plasmids.push(plasmid);
    }

    fn try_form_plasmid_bridge(&self, other_id: usize) -> Option<PlasmidPackage> {
        // check if THIS entity can form a plasmid bridge 
        if !self.plasmid_bridge {
            return None; // this entity cannot form a plasmid bridge
        }

        // get the oldest plasmid 
        let plasmid = if self.genome.plasmids.is_empty() {
            return None; // no plasmids to transfer
        } else {
            self.genome.plasmids[0] 
        };
        Some(PlasmidPackage {
            plasmid,
            receiver_id: other_id,
        })
    }


    pub(crate)fn resolve_collision(&mut self, space: &mut Space, entities: &Vec<Entity>) -> Option<PlasmidPackage> {

        // check for collisions with the space boundaries
        let collision = space.check_position(self.position.clone(), Some(self.size), Some(self.id), entities);

        let mut plasmid_package: Option<PlasmidPackage> = None;
        
        match collision {
            Collision::EntityCollision(other_velocity, mass, other_position, id) => {
                // check if the colliding entity is the last collided entity
                // this is used to avoid jittering
                if self.last_entity_collision.0 == id && self.last_entity_collision.1 > 0 {
                    // skip update if the entity just collided with the same entity
                    return plasmid_package;
                }

                plasmid_package = self.try_form_plasmid_bridge(id);

                // resolve collision with other entity
                // elastic collision resolution

                // source:
                // https://www.vobarian.com/collisions/2dcollisions2.pdf

                // Use the compressed formaula which directly computes the 2D vector
                // Formula described in https://www.youtube.com/watch?v=eED4bSkYCB8&t=1070s
                // This formula is essentially the same as the one in the pdf, but optimized for 2D vectors and 
                // puts the v_t and v_n calculations into a single dot product calculation

                let v1 = self.velocity.clone();
                let v2 = other_velocity.clone();
                

                // not including pi since it wasnt included in mass calculation of other entity
                let m1 = self.size.powi(2); /* * std::f32::consts::PI; */// mass of this entity
                let m2 = mass; // mass of the other entity

                let delta_v = &v1 - &v2;
                let delta_p = &self.position - other_position;


                let new_v1: Array1<f32> = v1 - (2.0 * m2 / (m1 + m2)) *
                 (delta_v.dot(&delta_p) / delta_p.dot(&delta_p)) * delta_p;
                                        // delta_p.dot(&delta_p) is the squared norm 
                
                self.velocity = new_v1;


                if self.velocity[0].is_nan() || self.velocity[1].is_nan() {
                    eprintln!("NaN velocity detected after collision resolution");
                    self.velocity = Array1::zeros(2);
                }

                self.last_entity_collision = (id, IDLE_COLLISION_TIMER);

            }


            Collision::BorderCollision(border) => {
                // check if the entity just collided with a border
                if self.last_border_collision > 0 {
                    return plasmid_package;
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
        plasmid_package
    }

    pub(crate) fn emit_ligands(&mut self, settings: &Settings) -> Vec<Ligand> {
        // Take the ligands from the entity and return them
        let ligands = self.ligands_to_emit.clone();
        self.ligands_to_emit.clear();
        // calculate energy cost 
        // always remove absolute energy of ligands  to prevent energy gain from negative energy ligands
        let energy_cost: f32 = ligands.iter().map(|x| x.energy.abs()).sum();

        // check settings to see if ligand emission is enabled
        if !settings.enable_entity_ligand_emission() {
            return vec![];
        }
        self.energy -= energy_cost;

        ligands
    }

    pub(crate) fn receive_ligand(&mut self, ligand: &Ligand, settings: &Settings) -> bool{
        // return true if the ligand was processed, false if it was ignored 

        // ignore self-emitted ligands (destroyed to prevent them from being stuck in the entity)
        if ligand.emitted_id == self.id {
            return true; 
        }

        if ligand.energy < 0.0 {
            self.receive_toxins(ligand.spec, settings);
            return true;
        }


        // process the ligand message
        // for now, just increase energy based on message

        let angle = calculate_ligand_direction(self, &ligand.position);

        // handle the message

        let angle_index = (angle / std::f64::consts::PI * (settings.receptors_per_entity() - 1) as f64).floor() as usize; // index in receptor array

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

        if index >= self.inner_protein_levels.len() {
            eprintln!("Concentration index {} out of bounds", index);
            return false; // invalid index
        }
        // change concentration and clamp to range
        self.inner_protein_levels[index] = (self.inner_protein_levels[index] + change).clamp(settings.concentration_range().0, settings.concentration_range().1);

        // add angle to received_ligands for statistics
        let angle: f32 = (angle_index as f32 / settings.receptors_per_entity() as f32) *180.0; // angle in degrees from 0 to 180
        self.received_ligands.push(angle as u8);

        return true;


    }

    #[cfg(feature = "cuda")]
    pub(crate) fn receive_ligand_cuda_shortcut(&mut self, spec: u16, receptor_index: usize, settings: &Settings) {
        // change energy

        let energy = crate::objects::ligand::get_ligand_energy(spec, settings);
        if energy < 0.0 {
            return self.receive_toxins(spec, settings);
        }


        use crate::objects::{OUTPUTS, receptor};
        self.energy += energy;

        // change concentration

        let (index, positive, _) = receptor::sequence_receptor(self.receptors[receptor_index]);

        let change: i16 = if positive { 1 } else { -1 };

        if index >= OUTPUTS as u8 {
            return; // invalid index
        }

        // change concentration and clamp to range
        self.inner_protein_levels[index as usize] = (self.inner_protein_levels[index as usize] + change).clamp(settings.concentration_range().0, settings.concentration_range().1);

        // add angle to received_ligands for statistics
        let angle: f32 = (receptor_index as f32 / settings.receptors_per_entity() as f32) *180.0; // angle in degrees from 0 to 180
        self.received_ligands.push(angle as u8);
    }

    pub(crate) fn receive_toxins(&mut self, spec: u16, settings: &Settings) {

        // check if any plasmid can disable it
        for &plasmid in &self.genome.plasmids {
            if plasmid == spec {
                // plasmid found, ignore negative ligand
                return;
            }
        }
        
        if settings.toxins_active() == false {
            eprintln!("Toxins are not active in settings");
            return ;
        }
        // decrease energy based on toxin level
        let energy = crate::objects::ligand::get_ligand_energy(spec, settings);
        self.energy -= energy.abs(); // always decrease energy
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
    }
}