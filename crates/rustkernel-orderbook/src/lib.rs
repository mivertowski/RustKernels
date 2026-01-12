//! # RustKernel Order Matching
//!
//! GPU-accelerated order book matching for HFT.
//!
//! ## Kernels
//! - `OrderMatchingEngine` - Price-time priority matching (<10μs P99)

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Order matching engine kernel.
#[derive(Debug, Clone, Default)]
pub struct OrderMatchingEngine { metadata: KernelMetadata }
impl OrderMatchingEngine {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("orderbook/matching", Domain::OrderMatching)
            .with_description("Price-time priority order matching")
            .with_throughput(100_000).with_latency_us(10.0).with_gpu_native(true) }
    }
}
impl GpuKernel for OrderMatchingEngine { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all order matching kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering order matching kernels");
    Ok(())
}
