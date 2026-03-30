use cpu_ai_framework::core;
use cpu_ai_framework::benchmark;

fn main() {
    println!("Next-Gen CPU AI Framework Initialized.");
    println!("Targeting 1.58-bit Ternary Neural Networks and Memory Sparse Attention (MSA).");

    let mut runtime = core::Runtime::new();
    runtime.initialize_memory_os();

    // Run the benchmarking suite to compare FP32 vs Ternary
    benchmark::run_benchmark();

    // Run the Toy Training Loop to prove 1.58-bit STE Learning
    benchmark::run_toy_training();
}
