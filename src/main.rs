use UniSim::prelude::*;

// Command matching loop buit with Copilot Agent




fn main() {

    let commands = vec![
        ("run <steps>", "Run the simulation by n steps."),
        ("run_seconds <seconds>", "Run the simulation for n seconds."),
        ("add_ligand_source <x, y, frequency>", "Add a ligand source, specifying its position and frequency."),
        ("delete_ligand_source <id>", "Deletes a ligand source"),
        ("delete_all_ligands", "Deletes all ligands in the world."),
        ("save <filename>", "Saves the current world state to a file."),
        ("cuda_init", "Initializes CUDA for GPU acceleration."),
        ("help", "Displays this help message."),
        ("spec_count", "Displays the count of each ligand type in the world."),
        ("exit", "Exits the simulation."),
    ];

    let args: Vec<String> = std::env::args().collect();
    
    let filename = if args.len() > 1 {
        args[1].clone()
    } else {
        println!("Please enter the filename:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).expect("Failed to read line");
        input.trim().to_string()
    };
    // Check if the file exists before proceeding
    let mut world = if !std::path::Path::new(&filename).exists() {
        println!("File '{}' does not exist.", filename);
        World::new(settings!())
    } else {
        
        World::new(settings!(filename))
    };




    loop{
        let mut input = String::new();
        println!("Enter command (type 'help' for commands): "); 
        std::io::stdin().read_line(&mut input).expect("Failed to read line");
        let command = input.trim();

        match command {
            "exit" => {
                println!("Exiting simulation.");
                world.close();
                break;
            },
            "help" => {
                println!("Available commands:");
                for (cmd, desc) in &commands {
                    println!("  {}: {}", cmd, desc);
                }
            },
            _ => {
                // Implement 'run <steps>' command
                if let Some(rest) = command.strip_prefix("run ") {
                    if let Ok(steps) = rest.trim().parse::<usize>() {
                        println!("Running simulation for {} steps...", steps);
                        world.run(steps);
                        continue;
                    } else {
                        println!("Invalid step count. Usage: run <steps>");
                        continue;
                    }
                }
                // Implement 'run_seconds <seconds>' command
                if let Some(rest) = command.strip_prefix("run_seconds ") {
                    if let Ok(seconds) = rest.trim().parse::<f32>() {
                        println!("Running simulation for {} seconds...", seconds);
                        world.run_seconds(seconds);
                        continue;
                    } else {
                        println!("Invalid seconds value. Usage: run_seconds <seconds>");
                        continue;
                    }
                }
                // Implement 'add_ligand_source <x, y, frequency>' command
                if let Some(rest) = command.strip_prefix("add_ligand_source") {
                    let parts: Vec<&str> = rest.trim().split(',').map(|s| s.trim()).collect();
                    if parts.len() == 3 {
                        let x = parts[0].parse::<f32>();
                        let y = parts[1].parse::<f32>();
                        let freq = parts[2].parse::<f32>();
                        if let (Ok(x), Ok(y), Ok(freq)) = (x, y, freq) {
                            match world.add_ligand_source(vec![x, y], freq, 0) { // Default ligand_spec=0, adjust if needed
                                Ok(_) => println!("Added ligand source at ({}, {}) with frequency {}.", x, y, freq),
                                Err(e) => println!("Failed to add ligand source: {}", e),
                            }
                        } else {
                            println!("Invalid arguments. Usage: add_ligand_source <x, y, frequency>");
                        }
                    } else {
                        println!("Invalid arguments. Usage: add_ligand_source <x, y, frequency>");
                    }
                    continue;
                }

                if command == "spec_count" {
                    let max_spec = world.settings.possible_ligands();
                    println!("Ligand type counts:");
                    for spec in 0..max_spec as u16 {
                        let count = world.get_spec_count(spec);
                        println!("  Spec {}: {}", spec, count);
                    }
                    continue;
                }
                #[cfg(feature = "cuda")]
                if command == "cuda_init" {
                    match world.cuda_initialize() {
                        Ok(_) => println!("CUDA initialized successfully."),
                        Err(e) => println!("Failed to initialize CUDA: {}", e),
                    }
                    continue;
                }
                #[cfg(not(feature = "cuda"))]
                if command == "cuda_init" {
                    println!("CUDA feature is not enabled in this build.");
                    continue;
                }
                if let Some(rest) = command.strip_prefix("save ") {
                    let filename = rest.trim();
                    if !filename.is_empty() {
                        match world.save(Some(filename), true) {
                            Ok(_) => println!("World saved to {}.", filename),
                            Err(e) => println!("Failed to save world: {}", e),
                        }
                    } else {
                        println!("Invalid filename. Usage: save <filename>");
                    }
                    continue;
                }
                if let Some(rest) = command.strip_prefix("delete_ligand_source ") {
                    let id = rest.trim().parse::<usize>();
                    match id {
                        Ok(idx) => match world.remove_ligand_source(idx) {
                            Ok(_) => println!("Deleted ligand source with id {}.", idx),
                            Err(e) => println!("Failed to delete ligand source: {}", e),
                        },
                        Err(_) => println!("Invalid id. Usage: delete_ligand_source <id>"),
                    }
                    continue;
                }
                if command == "delete_all_ligands" {
                    world.delete_all_ligands();
                    println!("All ligands deleted from the world.");
                    continue;
                }
                println!("Unknown command. Type 'help' for a list of commands.");
            }
        }
    }

    
}