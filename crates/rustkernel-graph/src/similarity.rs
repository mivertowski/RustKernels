//! Similarity measure kernels.
//!
//! - `JaccardSimilarity` - Set intersection over union
//! - `CosineSimilarity` - Dot product normalized
//! - `AdamicAdarIndex` - Weighted common neighbors

use rustkernel_core::{
    domain::Domain,
    kernel::KernelMetadata,
    traits::GpuKernel,
};

/// Jaccard similarity kernel.
#[derive(Debug, Clone)]
pub struct JaccardSimilarity {
    metadata: KernelMetadata,
}

impl JaccardSimilarity {
    /// Create a new Jaccard similarity kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/jaccard-similarity", Domain::GraphAnalytics)
                .with_description("Jaccard similarity: |A∩B|/|A∪B|")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }
}

impl Default for JaccardSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for JaccardSimilarity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Cosine similarity kernel.
#[derive(Debug, Clone)]
pub struct CosineSimilarity {
    metadata: KernelMetadata,
}

impl CosineSimilarity {
    /// Create a new cosine similarity kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/cosine-similarity", Domain::GraphAnalytics)
                .with_description("Cosine similarity: dot(A,B)/(|A|*|B|)")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }
}

impl Default for CosineSimilarity {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for CosineSimilarity {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Adamic-Adar index kernel.
#[derive(Debug, Clone)]
pub struct AdamicAdarIndex {
    metadata: KernelMetadata,
}

impl AdamicAdarIndex {
    /// Create a new Adamic-Adar index kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/adamic-adar-index", Domain::GraphAnalytics)
                .with_description("Adamic-Adar index: Σ 1/log(|N(z)|) for common neighbors")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }
}

impl Default for AdamicAdarIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for AdamicAdarIndex {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}
