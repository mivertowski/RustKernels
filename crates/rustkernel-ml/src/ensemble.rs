//! Ensemble method kernels.

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Ensemble voting kernel.
#[derive(Debug, Clone, Default)]
pub struct EnsembleVoting {
    metadata: KernelMetadata,
}

impl EnsembleVoting {
    /// Create a new ensemble voting kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/ensemble-voting", Domain::StatisticalML)
                .with_description("Weighted majority voting ensemble")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }
}

impl GpuKernel for EnsembleVoting {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}
