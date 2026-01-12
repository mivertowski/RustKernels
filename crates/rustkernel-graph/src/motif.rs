//! Motif detection kernels.
//!
//! - `TriangleCounting` - Ring kernel for triangle enumeration
//! - `MotifDetection` - Batch kernel for k-node subgraph census

use rustkernel_core::{
    domain::Domain,
    kernel::KernelMetadata,
    traits::GpuKernel,
};

/// Triangle counting kernel.
#[derive(Debug, Clone)]
pub struct TriangleCounting {
    metadata: KernelMetadata,
}

impl TriangleCounting {
    /// Create a new triangle counting kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("graph/triangle-counting", Domain::GraphAnalytics)
                .with_description("Local triangle enumeration")
                .with_throughput(500_000)
                .with_latency_us(0.5),
        }
    }
}

impl Default for TriangleCounting {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for TriangleCounting {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Motif detection kernel.
#[derive(Debug, Clone)]
pub struct MotifDetection {
    metadata: KernelMetadata,
}

impl MotifDetection {
    /// Create a new motif detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/motif-detection", Domain::GraphAnalytics)
                .with_description("k-node subgraph census")
                .with_throughput(1_000)
                .with_latency_us(10_000.0),
        }
    }
}

impl Default for MotifDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for MotifDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}
