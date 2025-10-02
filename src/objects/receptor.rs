use super::ReceptorBond;

pub fn bond(receptor: u32, message: u32) -> Option<ReceptorBond> {
    // returns: (energy change, (new receptor type, new receptor strength))
    if receptor == 0 {
        return None; // no receptor
    }

    let (inner_protein, positive, receptor_spec) = sequence_receptor(receptor);
    let (energy , message_spec) = sequence_message(message);

    let spec_match = (receptor_spec ^ message_spec).count_ones();

    // CONTINUEEEEE

    // example receptor types

    None
    
}

fn sequence_receptor(receptor: u32) -> (u8, bool, u16) {
    // returns the sequence of the receptor
    // first byte: type of inner protein
    // second byte (bool): negative or positive
    // last two bytes: specification number
    let bytes = receptor.to_le_bytes();
    (bytes[0], bytes[1] != 0, u16::from_le_bytes([bytes[2], bytes[3]]))

}

fn sequence_message(message: u32) -> (f32, u16) {
    // returns the sequence of the message
    let bytes = message.to_le_bytes();
    let energy = u16::from_le_bytes([bytes[0], bytes[1]]) as f32 / u16::MAX as f32; // energy in range 0..1

    (energy, u16::from_le_bytes([bytes[2], bytes[3]]))
}
