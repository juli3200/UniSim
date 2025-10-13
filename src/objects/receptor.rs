const SPEC_LENGTH: f64 = 16.0; // length of the specification in bits


fn spec_match(spec1: u16, spec2: u16) -> bool{
    let matches = (spec1 ^ spec2).count_ones();

    rand::random_bool(matches as f64 / SPEC_LENGTH)
}

pub fn bond(receptor: u32, message_spec: u16) -> Option<i32> {
    // returns: None if no bond formed
    // Some(inner_protein)
    if receptor == 0 {
        return None; // no receptor
    }

    let (inner_protein, positive, receptor_spec) = sequence_receptor(receptor);


    let bonding = spec_match(receptor_spec, message_spec);

    if !bonding {
        return None; // no bond formed
    }

    // inner protein negative if concentration should decrease
    // the abs value is the index
    let inner_protein: i32 =  if positive { inner_protein as i32 } else { -(inner_protein as i32) };

    return Some(inner_protein);
    
}

pub fn extract_receptor_fns(receptor_dna: u64) -> Box<dyn Fn(f32) -> f64> {
    // returns a function that takes in x value and returns a probability (0..1)
    // fn looks like: a*x^2 + b*x + c
    // a, b, c are derived from the receptor_dna

    let bytes = receptor_dna.to_le_bytes();

    let (mut a, mut b, c) = (bytes[1] as f32 / 512.0, bytes[2] as f32 / 512.0, bytes[3] as f32 / 512.0); // in range 0..0.5

    a -= 0.166;
    b -= 0.166;
    // c -= 0.0;


    // a, b are now in range -0.166..0.334
    // fn results are in range -0.33..1.333
    // c is not changed and is in range 0..0.5
    // this is to ensure that the function is not too extreme and overlaps 0 and one equally

    return Box::new( move |x: f32| {

                // quadratic function clamped to 0..1
                (a * x.powi(2) + b * x + c).clamp(0.0, 1.0) as f64

            }) as Box<dyn Fn(f32) -> f64>;

}

pub(crate)fn sequence_receptor(receptor: u32) -> (u8, bool, u16) {
    // returns the sequence of the receptor
    // first byte: type of inner protein
    // second byte (bool): negative or positive
    // last two bytes: specification number
    let bytes = receptor.to_le_bytes();
    (bytes[0], bytes[1] != 0, u16::from_le_bytes([bytes[2], bytes[3]]))

}




