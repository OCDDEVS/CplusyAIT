pub mod attention;
pub mod checkpoint;
pub mod mamba;
pub mod infer;

pub use candle_core::{Tensor, Device, DType};
pub use candle_nn::{Linear, Module};
use std::ffi::c_void;

/// The Core Orchestration Runtime
/// Thread pool orchestration, API entry point.
pub struct Runtime {
    pub thread_pool: rayon::ThreadPool,
    pub mapped_memory_pool: *mut c_void, // Pointer to mmaped NVMe storage
    pub preferred_device: Device,
}

impl Runtime {
    pub fn new() -> Self {
        // Initialize an optimal thread pool based on CPU cores
        let pool = rayon::ThreadPoolBuilder::new().build().unwrap();

        // Automatically detect and acquire best available compute device
        let preferred_device = Self::get_best_device();

        Runtime {
            thread_pool: pool,
            mapped_memory_pool: std::ptr::null_mut(),
            preferred_device,
        }
    }

    /// Detects if CUDA or Metal is available and returns the Device. Falls back to CPU.
    pub fn get_best_device() -> Device {
        if candle_core::utils::cuda_is_available() {
            println!("Runtime detected NVIDIA GPU (CUDA). Initializing heterogeneous pipeline.");
            return Device::new_cuda(0).unwrap_or(Device::Cpu);
        }

        if candle_core::utils::metal_is_available() {
            println!("Runtime detected Apple Silicon (Metal). Initializing heterogeneous pipeline.");
            return Device::new_metal(0).unwrap_or(Device::Cpu);
        }

        println!("No GPU detected. Running pure CPU optimized AVX2 pipeline.");
        Device::Cpu
    }

    /// Initializes the EverMemOS architecture for the local memory lifecycle.
    pub fn initialize_memory_os(&mut self) {
        println!("Initializing EverMemOS (Episodic Trace Formation -> Semantic Consolidation)");
    }
}

/// Safe wrapper for the AVX2 Fast Ternary GEMM C++ Kernel.
/// Accepts Candle Tensors, extracts their pointers, executes AVX2 SIMD, and returns a new Tensor.
pub fn run_avx2_ternary_gemm(
    packed_weights: &Tensor, // Shape (M, K/4) - uint8
    activations: &Tensor,    // Shape (K, N) - u8 (we cast to i8 internally since candle lacks i8)
    m: usize, k: usize, n: usize
) -> candle_core::Result<Tensor> {

    // Ensure tensors are contiguous on the CPU
    let w_cont = packed_weights.contiguous()?;
    let a_cont = activations.contiguous()?;

    let device = Device::Cpu;
    let mut output_vec = vec![0i32; m * n];

    // Extract raw pointers securely
    // Candle doesn't support i8 natively, so we pass as u8 and cast pointers to i8 for C++.
    let w_vec = w_cont.to_vec1::<u8>()?;
    let a_vec = a_cont.to_vec1::<u8>()?;

    unsafe {
        crate::ffi::ternary_gemm_avx2_packed(
            w_vec.as_ptr(),
            a_vec.as_ptr() as *const i8,
            output_vec.as_mut_ptr(),
            m, n, k
        );
    }

    // Convert back into a Candle Tensor
    Tensor::from_vec(output_vec, (m, n), &device)
}
pub mod train;
