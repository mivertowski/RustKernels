//! Community detection kernels.
//!
//! - `ModularityScore` - Batch kernel for modularity calculation
//! - `LouvainCommunity` - Batch kernel for Louvain community detection

use rustkernel_core::{
    domain::Domain,
    kernel::KernelMetadata,
    traits::GpuKernel,
};

/// Modularity score calculation kernel.
#[derive(Debug, Clone)]
pub struct ModularityScore {
    metadata: KernelMetadata,
}

impl ModularityScore {
    /// Create a new modularity score kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/modularity-score", Domain::GraphAnalytics)
                .with_description("Modularity score Q = (1/2m) * Σ[Aij - kikj/2m]δ(ci,cj)")
                .with_throughput(50_000)
                .with_latency_us(100.0),
        }
    }
}

impl Default for ModularityScore {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for ModularityScore {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Louvain community detection kernel.
#[derive(Debug, Clone)]
pub struct LouvainCommunity {
    metadata: KernelMetadata,
}

impl LouvainCommunity {
    /// Create a new Louvain community detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/louvain-community", Domain::GraphAnalytics)
                .with_description("Louvain community detection (multi-level optimization)")
                .with_throughput(10_000)
                .with_latency_us(1000.0),
        }
    }
}

impl Default for LouvainCommunity {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for LouvainCommunity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}
