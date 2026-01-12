//! Kernel metadata and configuration.
//!
//! This module defines the kernel execution modes and metadata structures
//! that mirror the C# `[GpuKernel]` attribute system.

use crate::domain::Domain;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Kernel execution mode.
///
/// Determines how the kernel is launched and manages state:
/// - `Batch`: Traditional GPU offload with CPU orchestration
/// - `Ring`: Persistent GPU-native actor with lock-free messaging
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelMode {
    /// Batch kernel mode (CPU-orchestrated).
    ///
    /// Characteristics:
    /// - 10-50μs launch overhead per invocation
    /// - State resides in CPU memory between operations
    /// - Data transfer via pinned memory (PCIe bandwidth limited)
    /// - Best for: Periodic heavy computation (>10ms compute time)
    Batch,

    /// Ring kernel mode (GPU-persistent actor).
    ///
    /// Characteristics:
    /// - 100-500ns message processing latency
    /// - State permanently in GPU memory
    /// - Communication via lock-free ring buffers (zero-copy)
    /// - Best for: High-frequency operations (>10K msgs/sec), real-time
    Ring,
}

impl KernelMode {
    /// Returns true if this is a batch kernel.
    #[must_use]
    pub const fn is_batch(&self) -> bool {
        matches!(self, KernelMode::Batch)
    }

    /// Returns true if this is a ring kernel.
    #[must_use]
    pub const fn is_ring(&self) -> bool {
        matches!(self, KernelMode::Ring)
    }

    /// Returns the typical launch overhead in microseconds.
    #[must_use]
    pub const fn typical_overhead_us(&self) -> f64 {
        match self {
            KernelMode::Batch => 30.0, // 10-50μs range, use 30 as typical
            KernelMode::Ring => 0.3,   // 100-500ns range, use 300ns as typical
        }
    }

    /// Returns the mode name as a string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            KernelMode::Batch => "batch",
            KernelMode::Ring => "ring",
        }
    }
}

impl fmt::Display for KernelMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Kernel metadata (mirrors C# `[GpuKernel]` attribute).
///
/// Contains all configuration and performance expectations for a kernel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelMetadata {
    /// Unique kernel identifier (e.g., "graph/pagerank").
    pub id: String,

    /// Kernel execution mode.
    pub mode: KernelMode,

    /// Business domain for licensing and organization.
    pub domain: Domain,

    /// Human-readable description.
    pub description: String,

    /// Expected throughput in operations per second.
    pub expected_throughput: u64,

    /// Target latency in microseconds.
    pub target_latency_us: f64,

    /// Whether this kernel requires GPU-native execution.
    ///
    /// If true, CPU fallback is not available.
    pub requires_gpu_native: bool,

    /// Version of the kernel implementation.
    pub version: u32,
}

impl KernelMetadata {
    /// Create a new batch kernel metadata.
    #[must_use]
    pub fn batch(id: impl Into<String>, domain: Domain) -> Self {
        Self {
            id: id.into(),
            mode: KernelMode::Batch,
            domain,
            description: String::new(),
            expected_throughput: 10_000,
            target_latency_us: 50.0,
            requires_gpu_native: false,
            version: 1,
        }
    }

    /// Create a new ring kernel metadata.
    #[must_use]
    pub fn ring(id: impl Into<String>, domain: Domain) -> Self {
        Self {
            id: id.into(),
            mode: KernelMode::Ring,
            domain,
            description: String::new(),
            expected_throughput: 100_000,
            target_latency_us: 1.0,
            requires_gpu_native: true,
            version: 1,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the expected throughput.
    #[must_use]
    pub fn with_throughput(mut self, ops_per_sec: u64) -> Self {
        self.expected_throughput = ops_per_sec;
        self
    }

    /// Set the target latency.
    #[must_use]
    pub fn with_latency_us(mut self, latency_us: f64) -> Self {
        self.target_latency_us = latency_us;
        self
    }

    /// Set whether GPU-native execution is required.
    #[must_use]
    pub fn with_gpu_native(mut self, required: bool) -> Self {
        self.requires_gpu_native = required;
        self
    }

    /// Set the version.
    #[must_use]
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Returns the feature string for licensing.
    ///
    /// Format: `Domain.KernelName` where KernelName is extracted from the ID.
    #[must_use]
    pub fn feature_string(&self) -> String {
        // Extract kernel name from ID (e.g., "graph/pagerank" -> "PageRank")
        let name = self.id.rsplit('/').next().unwrap_or(&self.id);
        let name = to_pascal_case(name);
        format!("{}.{}", self.domain, name)
    }

    /// Returns the full kernel ID path.
    #[must_use]
    pub fn full_id(&self) -> String {
        format!("{}/{}", self.domain.as_str().to_lowercase(), self.id)
    }
}

impl Default for KernelMetadata {
    fn default() -> Self {
        Self::batch("unnamed", Domain::Core)
    }
}

/// Convert a kebab-case or snake_case string to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split(|c| c == '-' || c == '_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// Builder for kernel metadata.
#[derive(Default)]
pub struct KernelMetadataBuilder {
    id: Option<String>,
    mode: Option<KernelMode>,
    domain: Option<Domain>,
    description: String,
    expected_throughput: u64,
    target_latency_us: f64,
    requires_gpu_native: bool,
    version: u32,
}

impl KernelMetadataBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_throughput: 10_000,
            target_latency_us: 50.0,
            version: 1,
            ..Default::default()
        }
    }

    /// Set the kernel ID.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the kernel mode.
    #[must_use]
    pub fn mode(mut self, mode: KernelMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Set the domain.
    #[must_use]
    pub fn domain(mut self, domain: Domain) -> Self {
        self.domain = Some(domain);
        self
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the expected throughput.
    #[must_use]
    pub fn throughput(mut self, ops_per_sec: u64) -> Self {
        self.expected_throughput = ops_per_sec;
        self
    }

    /// Set the target latency.
    #[must_use]
    pub fn latency_us(mut self, latency_us: f64) -> Self {
        self.target_latency_us = latency_us;
        self
    }

    /// Set whether GPU-native execution is required.
    #[must_use]
    pub fn gpu_native(mut self, required: bool) -> Self {
        self.requires_gpu_native = required;
        self
    }

    /// Set the version.
    #[must_use]
    pub fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Build the metadata.
    ///
    /// # Panics
    ///
    /// Panics if `id`, `mode`, or `domain` are not set.
    #[must_use]
    pub fn build(self) -> KernelMetadata {
        KernelMetadata {
            id: self.id.expect("id is required"),
            mode: self.mode.expect("mode is required"),
            domain: self.domain.expect("domain is required"),
            description: self.description,
            expected_throughput: self.expected_throughput,
            target_latency_us: self.target_latency_us,
            requires_gpu_native: self.requires_gpu_native,
            version: self.version,
        }
    }

    /// Try to build the metadata.
    ///
    /// Returns `None` if required fields are missing.
    #[must_use]
    pub fn try_build(self) -> Option<KernelMetadata> {
        Some(KernelMetadata {
            id: self.id?,
            mode: self.mode?,
            domain: self.domain?,
            description: self.description,
            expected_throughput: self.expected_throughput,
            target_latency_us: self.target_latency_us,
            requires_gpu_native: self.requires_gpu_native,
            version: self.version,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_mode_properties() {
        assert!(KernelMode::Batch.is_batch());
        assert!(!KernelMode::Batch.is_ring());
        assert!(KernelMode::Ring.is_ring());
        assert!(!KernelMode::Ring.is_batch());
    }

    #[test]
    fn test_kernel_metadata_batch() {
        let meta = KernelMetadata::batch("pagerank", Domain::GraphAnalytics)
            .with_description("PageRank centrality")
            .with_throughput(100_000)
            .with_latency_us(10.0);

        assert_eq!(meta.id, "pagerank");
        assert_eq!(meta.mode, KernelMode::Batch);
        assert_eq!(meta.domain, Domain::GraphAnalytics);
        assert!(!meta.requires_gpu_native);
    }

    #[test]
    fn test_kernel_metadata_ring() {
        let meta = KernelMetadata::ring("pagerank", Domain::GraphAnalytics);

        assert_eq!(meta.mode, KernelMode::Ring);
        assert!(meta.requires_gpu_native);
    }

    #[test]
    fn test_feature_string() {
        let meta = KernelMetadata::ring("pagerank", Domain::GraphAnalytics);
        assert_eq!(meta.feature_string(), "GraphAnalytics.Pagerank");

        let meta = KernelMetadata::ring("graph/degree-centrality", Domain::GraphAnalytics);
        assert_eq!(meta.feature_string(), "GraphAnalytics.DegreeCentrality");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("pagerank"), "Pagerank");
        assert_eq!(to_pascal_case("degree-centrality"), "DegreeCentrality");
        assert_eq!(to_pascal_case("snake_case"), "SnakeCase");
        assert_eq!(to_pascal_case("mixed-snake_case"), "MixedSnakeCase");
    }

    #[test]
    fn test_builder() {
        let meta = KernelMetadataBuilder::new()
            .id("test-kernel")
            .mode(KernelMode::Ring)
            .domain(Domain::Core)
            .throughput(50_000)
            .latency_us(0.5)
            .build();

        assert_eq!(meta.id, "test-kernel");
        assert_eq!(meta.mode, KernelMode::Ring);
        assert_eq!(meta.expected_throughput, 50_000);
    }
}
