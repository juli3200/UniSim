// fn that are not used anymore
// might be useful in the future

fn find_location(&self, slot: SaveSlot) -> Option<u32> {
        // clause if table was not called

        if self.iteration == 0 {
            return Some(HEADER_SIZE as u32);
        }
        

        fn inversive_find(iteration: u32, saved_states: u32, store_capacity: u32, location: u32, file: &mut File, jumper: bool) -> Option<u32> {

            if iteration == 1 {

                let jumper_address = location + saved_states % store_capacity * 4;
                // the jumper address is seeked return jumper
                if jumper { return Some(jumper_address); }

                let e1 = file.seek(SeekFrom::Start(jumper_address as u64));

                let mut buffer = [0u8; 4];
                let e2 = file.read_exact(&mut buffer);

                let state_location = u32::from_le_bytes(buffer);
                // the state location is seeked return state
                
                if e1.is_err() || e2.is_err() {
                    return None;
                }

                return Some(state_location);
            }

            let next_location;
            // use a block to delete variables(prevent stack overflow)
            {
                // opening the file at location
                let e1 = file.seek(SeekFrom::Start(location as u64));

                // reading the next location to the buffer
                let mut buffer = [0u8; 4];
                let e2 = file.read_exact(&mut buffer);

                // convert the buffer in a u32
                next_location = u32::from_le_bytes(buffer);

                // Error handling
                if e1.is_err() || e2.is_err() {
                    return None;
                }
            }   

            // recursive call
            inversive_find(iteration-1, saved_states, store_capacity, next_location, file, jumper)
            
        }

        let mut iteration;
        let target_slot;

        // jumper decides whether to look for a jumper or a state address
        let jumper;

        match slot {
            SaveSlot::Jumper(slot) => {
                iteration = (slot as f32 / self.settings.store_capacity as f32).ceil() as u32;
                target_slot = slot as u32;
                jumper = true;
            }
            SaveSlot::State(slot) => {
                iteration = (slot as f32 / self.settings.store_capacity as f32).ceil() as u32;
                target_slot = slot as u32;
                jumper = false;
            }
            SaveSlot::None => {
                iteration = self.iteration as u32;
                target_slot = self.saved_states as u32;
                jumper = true;
            }
        }

        if iteration == 0 {iteration = 1;} 

        // set location to default HEADER_SIZE
        let location = HEADER_SIZE as u32;

        // Error handled by returning None in case of failure
        if let Ok(mut file) = File::open(self.path.as_ref().unwrap()) {
            return inversive_find(iteration, target_slot, self.settings.store_capacity as u32, location, &mut file, jumper);
        }

        None
    }

    #[derive(Debug, Clone)]
pub(crate) enum SaveSlot{
    Jumper(usize),
    State(usize),
    None
}