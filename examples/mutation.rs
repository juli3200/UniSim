#[cfg(feature = "cuda")]
fn main() {
    use UniSim::prelude::*;
    
    let mut_rates = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2];
    let sigma_changes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];


    for i in 0..mut_rates.len() {

        let settings: Settings = settings!("experiments/mutation.json");

        let mut world = World::new(settings);

        edit_settings!(&mut world, mutation_rate = mut_rates[i], 
            path = format!("testfiles/mutation_rate_{}.bin", mut_rates[i]),
            std_dev_mutation = sigma_changes[i]
        );

        world.save(None, true).expect("Failed to save world");
        //world.cuda_initialize().expect("Failed to initialize CUDA");


        for i in 1..4{
            let _ = world.add_ligand_source(vec![50.0, 50.0], 5.0, i);
            world.run(2500);
            world.delete_all_ligands();
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Please enable the 'cuda' feature to run this example.");
}
