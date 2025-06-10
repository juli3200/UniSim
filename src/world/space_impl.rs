use super::*;

impl Space {
    // new

    pub(crate) fn empty() -> Self {
        // creates an empty space, only used as a placeholder
        Self {
            width: 0,
            height: 0,
            grid: Vec::new(),
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
}