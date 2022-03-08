//! Torch components
pub mod agents;
pub mod backends;
pub mod critic;
pub mod distributions;
pub mod features;
pub mod initializers;
pub mod modules;
pub mod optimizers;
pub mod updaters;
pub mod utils;

/// Initialization function for unit tests using CUDA.
///
/// Creating a Tensor on CUDA for the first time on each thread triggers a long (~1s) setup
/// routine.
/// Each test runs in its own thread so this setup happens many times (albeit concurrently).
/// With [`ctor`], this function automatically runs immediately before per-test threads are spawned
/// and creates a small CUDA tensor (if CUDA is available). Consequently, the initialization is
/// only performed once. The test threads spawn with the setup already in effect.
///
#[cfg(test)]
#[ctor::ctor]
fn cuda_test_init() {
    // Create a small CUDA tensor to trigger the slow CUDA setup routine.
    if tch::Cuda::is_available() {
        println!("Initializing CUDA for PyTorch before starting tests...");
        let _ = tch::Tensor::zeros(&[1], (tch::Kind::Int, tch::Device::Cuda(0)));
    }
}
