use rand::Rng;

use super::*;

impl Space {
    // new

    pub(crate) fn empty() -> Self {
        // creates an empty space, only used as a placeholder
        Self {
            width: 0,
            height: 0,
            grid: vec![],
            max_size: 0.0, // no entities or ligands, so max_size is 0
        }
    }

    pub(crate) fn new(settings: &Settings) -> Result<Self, String> {
        // creates a new space with the given width and height
        let (width, height) = settings.dimensions;
        let grid = vec![vec![Vec::new(); height as usize]; width as usize];
        let max_size = settings.spawn_size; // set max_size to the spawn size of entities
        if width == 0 || height == 0 {
            return Err("Invalid space dimensions".into());
        }
        Ok(Self {
            width,
            height,
            grid,
            max_size
        })
    }

    pub(crate) fn get_random_position(&self, size: f32) -> Result<Array1<f32>, String> {
        // returns a random position in the space that is not occupied by any entity or ligand
        // size is the size of the entity or ligand

        let mut rng = rand::rng();
        let mut c = 0;
        let (x, y) = loop {
            let x = rng.random_range(size..(self.width as f32 - size));
            let y = rng.random_range(size..(self.height as f32 - size));
            if let Collision::NoCollision = self.check_position(Array1::from_vec(vec![x, y]), Some(size), None) {
                break (x, y);
            }
            c += 1;
            if c > 10000 {
                return Err(format!("Failed to find a random position in space after 10000 attempts, size: {}", size));
            }
        };
        Ok(Array1::from_vec(vec![x, y]))

    }


    pub(crate) fn check_position(&self, position: Array1<f32>, size: Option<f32>, id : Option<usize>) -> Collision {
        // checks if the position is valid in the space
        // position is a tuple of (x, y)
        // size is the size of the entity 
        // ligand has no size size is = 0


        // extract size or set to 0 if None
        // if size is None, it means the object is a ligand which has no size
        let size = if let Some(s) = size {
            s
        } else {
            0.0
        };

        // check if position is within the bounds of the space
        // returns Collision::BorderCollision if the position is out of bounds
        let (width, height) = (self.width as f32, self.height as f32);
        if position[0] - size < 0.0 {
            return Collision::BorderCollision(Border::Left);
        }
        if position[0] + size >= width {
            return Collision::BorderCollision(Border::Right);
        }
        if position[1] - size < 0.0 {
            return Collision::BorderCollision(Border::Top);
        }
        if position[1] + size >= height {
            return Collision::BorderCollision(Border::Bottom);
        }

        // ensure max_size is at least as large as size
        let max_size = self.max_size.max(size);
        // check if position is occupied by any entity or object

        // iterate over the grid cells that are within the max_size range of the position
        // this is done to avoid checking every single object in the space
        for x in (position[0] - max_size as f32).floor() as u32..(position[0] + max_size as f32).ceil() as u32 {
            for y in (position[1] - max_size as f32).floor() as u32..(position[1] + max_size as f32).ceil() as u32 {

                // get objects from the  grid
                let objects = &self.grid[x as usize][y as usize];
                
                for object in objects {
                    match object {
                        objects::ObjectType::Entity(entity) => {
                            let entity_ref = entity.borrow();
                            
                            
                            if let Some(id) = id {
                                if entity_ref.id == id {
                                    continue; // skip the entity with the given id, because it is itself
                                }
                            }

                            // dist is distance between the entity and the position
                            // Use squared distance to avoid unnecessary sqrt calculation
                            let dx = entity_ref.position[0] - position[0];
                            let dy = entity_ref.position[1] - position[1];
                            let min_dist_sq = (entity_ref.size + size).powi(2);
                            let dist_sq = dx * dx + dy * dy;
                            if dist_sq < min_dist_sq {
                                return Collision::EntityCollision(entity_ref.velocity.clone(), entity_ref.size.powi(2) /* std::f32::consts::PI*/, entity_ref.position.clone()); // return the velocity and mass of the coll
                            }
                        }
                        _ => {}
                    }
                }
            
            }
        }
        Collision::NoCollision // position is valid
    }
}