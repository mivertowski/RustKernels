//! # RustKernel Payment Processing
//!
//! GPU-accelerated payment processing kernels.
//!
//! ## Kernels
//! - `PaymentProcessing` - Transaction execution
//! - `FlowAnalysis` - Payment flow metrics

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Payment processing kernel.
#[derive(Debug, Clone, Default)]
pub struct PaymentProcessing { metadata: KernelMetadata }
impl PaymentProcessing {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("payments/processing", Domain::PaymentProcessing)
            .with_description("Payment transaction execution")
            .with_throughput(100_000).with_latency_us(10.0) }
    }
}
impl GpuKernel for PaymentProcessing { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Payment flow analysis kernel.
#[derive(Debug, Clone, Default)]
pub struct FlowAnalysis { metadata: KernelMetadata }
impl FlowAnalysis {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("payments/flow-analysis", Domain::PaymentProcessing)
            .with_description("Payment flow metrics analysis")
            .with_throughput(50_000).with_latency_us(50.0) }
    }
}
impl GpuKernel for FlowAnalysis { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all payment kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering payment processing kernels");
    Ok(())
}
