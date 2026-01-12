//! Anomaly detection kernels.

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Isolation Forest kernel.
#[derive(Debug, Clone, Default)]
pub struct IsolationForest {
    metadata: KernelMetadata,
}

impl IsolationForest {
    /// Create a new Isolation Forest kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/isolation-forest", Domain::StatisticalML)
                .with_description("Isolation Forest ensemble anomaly detection")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }
}

impl GpuKernel for IsolationForest {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}

/// Local Outlier Factor kernel.
#[derive(Debug, Clone, Default)]
pub struct LocalOutlierFactor {
    metadata: KernelMetadata,
}

impl LocalOutlierFactor {
    /// Create a new LOF kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/local-outlier-factor", Domain::StatisticalML)
                .with_description("Local Outlier Factor (k-NN density estimation)")
                .with_throughput(5_000)
                .with_latency_us(200.0),
        }
    }
}

impl GpuKernel for LocalOutlierFactor {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}
