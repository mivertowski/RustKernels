//! Clustering kernels.

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// K-Means clustering kernel.
#[derive(Debug, Clone, Default)]
pub struct KMeans {
    metadata: KernelMetadata,
}

impl KMeans {
    /// Create a new K-Means kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/kmeans-cluster", Domain::StatisticalML)
                .with_description("K-Means clustering with K-Means++ initialization")
                .with_throughput(20_000)
                .with_latency_us(50.0),
        }
    }
}

impl GpuKernel for KMeans {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}

/// DBSCAN clustering kernel.
#[derive(Debug, Clone, Default)]
pub struct DBSCAN {
    metadata: KernelMetadata,
}

impl DBSCAN {
    /// Create a new DBSCAN kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/dbscan-cluster", Domain::StatisticalML)
                .with_description("Density-based clustering with GPU union-find")
                .with_throughput(1_000)
                .with_latency_us(10_000.0),
        }
    }
}

impl GpuKernel for DBSCAN {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}

/// Hierarchical clustering kernel.
#[derive(Debug, Clone, Default)]
pub struct HierarchicalClustering {
    metadata: KernelMetadata,
}

impl HierarchicalClustering {
    /// Create a new hierarchical clustering kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/hierarchical-cluster", Domain::StatisticalML)
                .with_description("Agglomerative hierarchical clustering")
                .with_throughput(500)
                .with_latency_us(50_000.0),
        }
    }
}

impl GpuKernel for HierarchicalClustering {
    fn metadata(&self) -> &KernelMetadata { &self.metadata }
}
