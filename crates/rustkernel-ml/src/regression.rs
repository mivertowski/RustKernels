//! Regression kernels.

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Linear regression kernel.
#[derive(Debug, Clone, Default)]
pub struct LinearRegression {
    metadata: KernelMetadata,
}

impl LinearRegression {
    /// Create a new linear regression kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/linear-regression", Domain::StatisticalML)
                .with_description("OLS linear regression via normal equations")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }
}

impl GpuKernel for LinearRegression {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}

/// Ridge regression kernel.
#[derive(Debug, Clone, Default)]
pub struct RidgeRegression {
    metadata: KernelMetadata,
}

impl RidgeRegression {
    /// Create a new ridge regression kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/ridge-regression", Domain::StatisticalML)
                .with_description("Ridge regression with L2 regularization")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }
}

impl GpuKernel for RidgeRegression {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}
