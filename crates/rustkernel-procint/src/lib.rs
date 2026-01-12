//! # RustKernel Process Intelligence
//!
//! GPU-accelerated process mining and conformance checking.
//!
//! ## Kernels
//! - `DFGConstruction` - Directly-follows graph construction
//! - `PartialOrderAnalysis` - Concurrency detection
//! - `ConformanceChecking` - Multi-model conformance (DFG/Petri/BPMN)
//! - `OCPMPatternMatching` - Object-centric process mining

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// DFG construction kernel.
#[derive(Debug, Clone, Default)]
pub struct DFGConstruction { metadata: KernelMetadata }
impl DFGConstruction {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("procint/dfg-construction", Domain::ProcessIntelligence)
            .with_description("Directly-follows graph construction")
            .with_throughput(100_000).with_latency_us(50.0) }
    }
}
impl GpuKernel for DFGConstruction { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Conformance checking kernel.
#[derive(Debug, Clone, Default)]
pub struct ConformanceChecking { metadata: KernelMetadata }
impl ConformanceChecking {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::ring("procint/conformance-checking", Domain::ProcessIntelligence)
            .with_description("Multi-model conformance checking")
            .with_throughput(50_000).with_latency_us(100.0) }
    }
}
impl GpuKernel for ConformanceChecking { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all process intelligence kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering process intelligence kernels");
    Ok(())
}
