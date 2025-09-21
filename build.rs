// build file is used to link all cuda libs

#[cfg(feature = "cuda")]
const NAMES: [&str; 3] = ["test", "memory", "grid"];

#[cfg(target_os = "linux")]
const CUDA_PATH: &str = "/usr/local/cuda/lib64";
#[cfg(not(target_os = "linux"))]
const CUDA_PATH: &str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64";

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
    println!(r"cargo:rustc-link-search={}", CUDA_PATH);

    // link all libs that are used in the project
    for name in NAMES {
        println!("cargo:rustc-link-lib=static={}", name);
    }

    // Search path for cuda libraries
    println!(r"cargo:rustc-link-search=.\lib");

    }
    
}