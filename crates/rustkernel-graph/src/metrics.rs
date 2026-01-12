//! Graph metric kernels.
//!
//! - `GraphDensity` - Edge density calculation
//! - `ClusteringCoefficient` - Local and global clustering

use rustkernel_core::{
    domain::Domain,
    kernel::KernelMetadata,
    traits::GpuKernel,
};

/// Graph density kernel.
#[derive(Debug, Clone)]
pub struct GraphDensity {
    metadata: KernelMetadata,
}

impl GraphDensity {
    /// Create a new graph density kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/density", Domain::GraphAnalytics)
                .with_description("Graph density: 2E/(V*(V-1))")
                .with_throughput(1_000_000)
                .with_latency_us(1.0),
        }
    }
}

impl Default for GraphDensity {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for GraphDensity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Average path length kernel.
#[derive(Debug, Clone)]
pub struct AveragePathLength {
    metadata: KernelMetadata,
}

impl AveragePathLength {
    /// Create a new average path length kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/average-path-length", Domain::GraphAnalytics)
                .with_description("Average shortest path length (Floyd-Warshall/BFS)")
                .with_throughput(1_000)
                .with_latency_us(10_000.0),
        }
    }
}

impl Default for AveragePathLength {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for AveragePathLength {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Clustering coefficient kernel.
#[derive(Debug, Clone)]
pub struct ClusteringCoefficient {
    metadata: KernelMetadata,
}

impl ClusteringCoefficient {
    /// Create a new clustering coefficient kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/clustering-coefficient", Domain::GraphAnalytics)
                .with_description("Local and global clustering coefficient")
                .with_throughput(50_000)
                .with_latency_us(50.0),
        }
    }
}

impl Default for ClusteringCoefficient {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for ClusteringCoefficient {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}
