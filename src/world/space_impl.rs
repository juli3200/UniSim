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
        }
    }

    pub(crate) fn new(dim: (u32, u32)) -> Result<Self, String> {
        // creates a new space with the given width and height
        let (width, height) = dim;
        let grid = vec![vec![Vec::new(); height as usize]; width as usize];
        if width == 0 || height == 0 {
            return Err("Invalid space dimensions".into());
        }
        Ok(Self {
            width,
            height,
            grid,
        })
    }

    pub(crate) fn get_random_position(&self, size: f32) -> Option<(f32, f32)> {
        // returns a random position in the space that is not occupied by any entity or ligand
        // size is the size of the entity or ligand

        let mut rng = rand::rng();
        let x = rng.random_range(0.0..self.width as f32 - size);
        let y = rng.random_range(0.0..self.height as f32 - size);
        Some((x, y))
    }


    pub(crate) fn check_position(&self, position: (f32, f32), size: Option<f32>) -> bool {
        // checks if the position is valid in the space
        // position is a tuple of (x, y)
        // size is the size of the entity 
        // ligand has no size size is = 0



        let size = if let Some(s) = size {
            s
        } else {
            0.0
        };

        let (width, height) = (self.width as f32, self.height as f32);
        if position.0 - size < 0.0 || position.0 + size >= width || position.1 - size < 0.0 || position.1 + size >= height {
            return false; // out of bounds
        }

        

        // check of position is occupied by any entity or object
        true
    }
}