fn main() {
    // Compile the C++ compute kernels
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let mut builder = cc::Build::new();

    builder.cpp(true)
        .file("kernels/fp32_gemm.cpp")
        .file("kernels/cpu_ternary_gemm.cpp")
        .file("kernels/msa_router.cpp")
        .file("kernels/memory_paging.cpp")
        .flag_if_supported("-O3")
        .flag_if_supported("-std=c++17");

    // Only compile the AVX2 SIMD kernel if the target architecture is x86_64
    if target_arch == "x86_64" {
        builder.file("kernels/cpu_ternary_gemm_avx2.cpp")
               .flag_if_supported("-mavx2");
    }

    builder.compile("cpu_ai_kernels");

    println!("cargo:rerun-if-changed=kernels/fp32_gemm.cpp");
    println!("cargo:rerun-if-changed=kernels/cpu_ternary_gemm.cpp");
    println!("cargo:rerun-if-changed=kernels/msa_router.cpp");
    println!("cargo:rerun-if-changed=kernels/memory_paging.cpp");
    if target_arch == "x86_64" {
        println!("cargo:rerun-if-changed=kernels/cpu_ternary_gemm_avx2.cpp");
    }
}
