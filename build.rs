// build file is used to link all cuda libs

const NAMES: [&str; 1] = ["test"];

fn main() {

    // only execute if CUDA feature is enabled
    #[cfg(feature = "cuda")]
    {
        // Link the CUDA libraries
        println!("cargo:rustc-link-lib=static=cuda");
        println!("cargo:rustc-link-lib=static=cudart");
        println!("cargo:rustc-link-lib=static=cublas");
        println!("cargo:rustc-link-lib=static=curand");
    
    // Add the path to the CUDA libraries
    // Adjust the path according to your CUDA installation
    // todo: make this dynamic
    println!(r"cargo:rustc-link-search=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64");

    // link all libs that are used in the project
    for name in NAMES {
        println!("cargo:rustc-link-lib=static={}", name);
    }

    // Search path for cuda libraries
    println!(r"cargo:rustc-link-search=.\lib");

    }
    
}