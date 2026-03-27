fn main() {
    // Compile the C++ compute kernels
    cc::Build::new()
        .cpp(true)
        .file("kernels/cpu_ternary_gemm.cpp")
        .file("kernels/msa_router.cpp")
        .file("kernels/memory_paging.cpp")
        // Use standard optimization flags; in a real scenario we'd target specific SIMD features
        // e.g., .flag("-mavx512f") depending on the host architecture
        .flag_if_supported("-O3")
        .flag_if_supported("-std=c++17")
        .compile("cpu_ai_kernels");

    // Tell cargo to invalidate the built crate whenever the C++ code changes
    println!("cargo:rerun-if-changed=kernels/cpu_ternary_gemm.cpp");
    println!("cargo:rerun-if-changed=kernels/msa_router.cpp");
    println!("cargo:rerun-if-changed=kernels/memory_paging.cpp");
}
