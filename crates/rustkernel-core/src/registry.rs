//! Kernel registry with auto-discovery.
//!
//! The registry manages all registered kernels and provides lookup functionality.
//! Kernels can be registered manually or discovered automatically via proc macros.

use crate::domain::Domain;
use crate::error::{KernelError, Result};
use crate::kernel::{KernelMetadata, KernelMode};
use crate::license::{LicenseError, LicenseValidator, SharedLicenseValidator};
use crate::traits::{BatchKernelDyn, RingKernelDyn};
use hashbrown::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{debug, info, warn};

/// Registry statistics.
#[derive(Debug, Clone, Default)]
pub struct RegistryStats {
    /// Total number of registered kernels.
    pub total: usize,
    /// Number of batch kernels.
    pub batch_kernels: usize,
    /// Number of ring kernels.
    pub ring_kernels: usize,
    /// Kernels by domain.
    pub by_domain: HashMap<Domain, usize>,
}

/// Entry for a batch kernel in the registry.
#[derive(Clone)]
pub struct BatchKernelEntry {
    /// Kernel metadata.
    pub metadata: KernelMetadata,
    /// Factory function to create the kernel.
    factory: Arc<dyn Fn() -> Arc<dyn BatchKernelDyn> + Send + Sync>,
}

impl BatchKernelEntry {
    /// Create a new batch kernel entry.
    pub fn new<F>(metadata: KernelMetadata, factory: F) -> Self
    where
        F: Fn() -> Arc<dyn BatchKernelDyn> + Send + Sync + 'static,
    {
        Self {
            metadata,
            factory: Arc::new(factory),
        }
    }

    /// Create an instance of the kernel.
    #[must_use]
    pub fn create(&self) -> Arc<dyn BatchKernelDyn> {
        (self.factory)()
    }
}

impl std::fmt::Debug for BatchKernelEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchKernelEntry")
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Entry for a ring kernel in the registry.
#[derive(Clone)]
pub struct RingKernelEntry {
    /// Kernel metadata.
    pub metadata: KernelMetadata,
    /// Factory function to create the kernel.
    factory: Arc<dyn Fn() -> Arc<dyn RingKernelDyn> + Send + Sync>,
}

impl RingKernelEntry {
    /// Create a new ring kernel entry.
    pub fn new<F>(metadata: KernelMetadata, factory: F) -> Self
    where
        F: Fn() -> Arc<dyn RingKernelDyn> + Send + Sync + 'static,
    {
        Self {
            metadata,
            factory: Arc::new(factory),
        }
    }

    /// Create an instance of the kernel.
    #[must_use]
    pub fn create(&self) -> Arc<dyn RingKernelDyn> {
        (self.factory)()
    }
}

impl std::fmt::Debug for RingKernelEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingKernelEntry")
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Central registry for all kernels.
#[derive(Debug)]
pub struct KernelRegistry {
    /// Batch kernel entries by ID.
    batch_kernels: RwLock<HashMap<String, BatchKernelEntry>>,
    /// Ring kernel entries by ID.
    ring_kernels: RwLock<HashMap<String, RingKernelEntry>>,
    /// Metadata-only entries (for kernels without type-erased factories).
    metadata_only: RwLock<HashMap<String, KernelMetadata>>,
    /// License validator.
    license: Option<SharedLicenseValidator>,
}

impl KernelRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            batch_kernels: RwLock::new(HashMap::new()),
            ring_kernels: RwLock::new(HashMap::new()),
            metadata_only: RwLock::new(HashMap::new()),
            license: None,
        }
    }

    /// Create a registry with license validation.
    #[must_use]
    pub fn with_license(license: SharedLicenseValidator) -> Self {
        Self {
            batch_kernels: RwLock::new(HashMap::new()),
            ring_kernels: RwLock::new(HashMap::new()),
            metadata_only: RwLock::new(HashMap::new()),
            license: Some(license),
        }
    }

    /// Set the license validator.
    pub fn set_license(&mut self, license: SharedLicenseValidator) {
        self.license = Some(license);
    }

    /// Register a batch kernel.
    pub fn register_batch(&self, entry: BatchKernelEntry) -> Result<()> {
        // Validate license if present
        if let Some(ref license) = self.license {
            self.validate_kernel_license(license.as_ref(), &entry.metadata)?;
        }

        let id = entry.metadata.id.clone();
        let mut kernels = self.batch_kernels.write().unwrap();

        if kernels.contains_key(&id) {
            return Err(KernelError::KernelAlreadyRegistered(id));
        }

        debug!(kernel_id = %id, domain = %entry.metadata.domain, "Registering batch kernel");
        kernels.insert(id, entry);
        Ok(())
    }

    /// Register a ring kernel.
    pub fn register_ring(&self, entry: RingKernelEntry) -> Result<()> {
        // Validate license if present
        if let Some(ref license) = self.license {
            self.validate_kernel_license(license.as_ref(), &entry.metadata)?;
        }

        let id = entry.metadata.id.clone();
        let mut kernels = self.ring_kernels.write().unwrap();

        if kernels.contains_key(&id) {
            return Err(KernelError::KernelAlreadyRegistered(id));
        }

        debug!(kernel_id = %id, domain = %entry.metadata.domain, "Registering ring kernel");
        kernels.insert(id, entry);
        Ok(())
    }

    /// Register kernel metadata without a factory.
    ///
    /// This is useful for kernels that implement `GpuKernel` but not the type-erased
    /// `BatchKernelDyn` or `RingKernelDyn` traits. The metadata is stored for discovery
    /// purposes, but the kernel cannot be instantiated through the registry.
    pub fn register_metadata(&self, metadata: KernelMetadata) -> Result<()> {
        // Validate license if present
        if let Some(ref license) = self.license {
            self.validate_kernel_license(license.as_ref(), &metadata)?;
        }

        let id = metadata.id.clone();

        // Check if already registered in any category
        {
            let batch = self.batch_kernels.read().unwrap();
            if batch.contains_key(&id) {
                return Err(KernelError::KernelAlreadyRegistered(id));
            }
        }
        {
            let ring = self.ring_kernels.read().unwrap();
            if ring.contains_key(&id) {
                return Err(KernelError::KernelAlreadyRegistered(id));
            }
        }

        let mut metadata_map = self.metadata_only.write().unwrap();
        if metadata_map.contains_key(&id) {
            return Err(KernelError::KernelAlreadyRegistered(id));
        }

        debug!(kernel_id = %id, domain = %metadata.domain, mode = ?metadata.mode, "Registering kernel metadata");
        metadata_map.insert(id, metadata);
        Ok(())
    }

    /// Validate kernel license.
    fn validate_kernel_license(
        &self,
        license: &dyn LicenseValidator,
        metadata: &KernelMetadata,
    ) -> Result<()> {
        // Validate domain
        license
            .validate_domain(metadata.domain)
            .map_err(KernelError::from)?;

        // Validate GPU-native requirement
        if metadata.requires_gpu_native && !license.gpu_native_enabled() {
            return Err(KernelError::from(LicenseError::GpuNativeNotLicensed));
        }

        Ok(())
    }

    /// Get a batch kernel by ID.
    #[must_use]
    pub fn get_batch(&self, id: &str) -> Option<BatchKernelEntry> {
        let kernels = self.batch_kernels.read().unwrap();
        kernels.get(id).cloned()
    }

    /// Get a ring kernel by ID.
    #[must_use]
    pub fn get_ring(&self, id: &str) -> Option<RingKernelEntry> {
        let kernels = self.ring_kernels.read().unwrap();
        kernels.get(id).cloned()
    }

    /// Get any kernel metadata by ID (batch, ring, or metadata-only).
    #[must_use]
    pub fn get(&self, id: &str) -> Option<KernelMetadata> {
        if let Some(entry) = self.get_batch(id) {
            return Some(entry.metadata);
        }
        if let Some(entry) = self.get_ring(id) {
            return Some(entry.metadata);
        }
        let metadata_map = self.metadata_only.read().unwrap();
        metadata_map.get(id).cloned()
    }

    /// Get metadata-only entry by ID.
    #[must_use]
    pub fn get_metadata_only(&self, id: &str) -> Option<KernelMetadata> {
        let metadata_map = self.metadata_only.read().unwrap();
        metadata_map.get(id).cloned()
    }

    /// Check if a kernel exists.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        let batch = self.batch_kernels.read().unwrap();
        let ring = self.ring_kernels.read().unwrap();
        let metadata = self.metadata_only.read().unwrap();
        batch.contains_key(id) || ring.contains_key(id) || metadata.contains_key(id)
    }

    /// Get all batch kernel IDs.
    #[must_use]
    pub fn batch_kernel_ids(&self) -> Vec<String> {
        let kernels = self.batch_kernels.read().unwrap();
        kernels.keys().cloned().collect()
    }

    /// Get all ring kernel IDs.
    #[must_use]
    pub fn ring_kernel_ids(&self) -> Vec<String> {
        let kernels = self.ring_kernels.read().unwrap();
        kernels.keys().cloned().collect()
    }

    /// Get all metadata-only kernel IDs.
    #[must_use]
    pub fn metadata_only_ids(&self) -> Vec<String> {
        let metadata = self.metadata_only.read().unwrap();
        metadata.keys().cloned().collect()
    }

    /// Get all kernel IDs.
    #[must_use]
    pub fn all_kernel_ids(&self) -> Vec<String> {
        let mut ids = self.batch_kernel_ids();
        ids.extend(self.ring_kernel_ids());
        ids.extend(self.metadata_only_ids());
        ids
    }

    /// Get kernels by domain.
    #[must_use]
    pub fn by_domain(&self, domain: Domain) -> Vec<KernelMetadata> {
        let mut result = Vec::new();

        let batch = self.batch_kernels.read().unwrap();
        for entry in batch.values() {
            if entry.metadata.domain == domain {
                result.push(entry.metadata.clone());
            }
        }

        let ring = self.ring_kernels.read().unwrap();
        for entry in ring.values() {
            if entry.metadata.domain == domain {
                result.push(entry.metadata.clone());
            }
        }

        let metadata = self.metadata_only.read().unwrap();
        for entry in metadata.values() {
            if entry.domain == domain {
                result.push(entry.clone());
            }
        }

        result
    }

    /// Get kernels by mode.
    #[must_use]
    pub fn by_mode(&self, mode: KernelMode) -> Vec<KernelMetadata> {
        let mut result: Vec<KernelMetadata> = match mode {
            KernelMode::Batch => {
                let kernels = self.batch_kernels.read().unwrap();
                kernels.values().map(|e| e.metadata.clone()).collect()
            }
            KernelMode::Ring => {
                let kernels = self.ring_kernels.read().unwrap();
                kernels.values().map(|e| e.metadata.clone()).collect()
            }
        };

        // Include metadata-only entries with matching mode
        let metadata = self.metadata_only.read().unwrap();
        for entry in metadata.values() {
            if entry.mode == mode {
                result.push(entry.clone());
            }
        }

        result
    }

    /// Get registry statistics.
    #[must_use]
    pub fn stats(&self) -> RegistryStats {
        let batch = self.batch_kernels.read().unwrap();
        let ring = self.ring_kernels.read().unwrap();
        let metadata = self.metadata_only.read().unwrap();

        let mut by_domain: HashMap<Domain, usize> = HashMap::new();

        for entry in batch.values() {
            *by_domain.entry(entry.metadata.domain).or_default() += 1;
        }

        for entry in ring.values() {
            *by_domain.entry(entry.metadata.domain).or_default() += 1;
        }

        // Count metadata-only entries by their mode for batch/ring stats
        let mut metadata_batch = 0;
        let mut metadata_ring = 0;
        for entry in metadata.values() {
            *by_domain.entry(entry.domain).or_default() += 1;
            match entry.mode {
                KernelMode::Batch => metadata_batch += 1,
                KernelMode::Ring => metadata_ring += 1,
            }
        }

        RegistryStats {
            total: batch.len() + ring.len() + metadata.len(),
            batch_kernels: batch.len() + metadata_batch,
            ring_kernels: ring.len() + metadata_ring,
            by_domain,
        }
    }

    /// Total number of registered kernels.
    #[must_use]
    pub fn total_count(&self) -> usize {
        let batch = self.batch_kernels.read().unwrap();
        let ring = self.ring_kernels.read().unwrap();
        let metadata = self.metadata_only.read().unwrap();
        batch.len() + ring.len() + metadata.len()
    }

    /// Clear all registered kernels.
    pub fn clear(&self) {
        let mut batch = self.batch_kernels.write().unwrap();
        let mut ring = self.ring_kernels.write().unwrap();
        let mut metadata = self.metadata_only.write().unwrap();
        batch.clear();
        ring.clear();
        metadata.clear();
        info!("Cleared kernel registry");
    }

    /// Unregister a kernel by ID.
    pub fn unregister(&self, id: &str) -> bool {
        let mut batch = self.batch_kernels.write().unwrap();
        if batch.remove(id).is_some() {
            debug!(kernel_id = %id, "Unregistered batch kernel");
            return true;
        }

        let mut ring = self.ring_kernels.write().unwrap();
        if ring.remove(id).is_some() {
            debug!(kernel_id = %id, "Unregistered ring kernel");
            return true;
        }

        let mut metadata = self.metadata_only.write().unwrap();
        if metadata.remove(id).is_some() {
            debug!(kernel_id = %id, "Unregistered metadata-only kernel");
            return true;
        }

        warn!(kernel_id = %id, "Attempted to unregister non-existent kernel");
        false
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for kernel registry.
#[derive(Default)]
pub struct KernelRegistryBuilder {
    license: Option<SharedLicenseValidator>,
    batch_entries: Vec<BatchKernelEntry>,
    ring_entries: Vec<RingKernelEntry>,
}

impl KernelRegistryBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the license validator.
    #[must_use]
    pub fn with_license(mut self, license: SharedLicenseValidator) -> Self {
        self.license = Some(license);
        self
    }

    /// Add a batch kernel.
    #[must_use]
    pub fn with_batch(mut self, entry: BatchKernelEntry) -> Self {
        self.batch_entries.push(entry);
        self
    }

    /// Add a ring kernel.
    #[must_use]
    pub fn with_ring(mut self, entry: RingKernelEntry) -> Self {
        self.ring_entries.push(entry);
        self
    }

    /// Build the registry.
    ///
    /// # Errors
    ///
    /// Returns an error if any kernel fails license validation.
    pub fn build(self) -> Result<KernelRegistry> {
        let registry = match self.license {
            Some(license) => KernelRegistry::with_license(license),
            None => KernelRegistry::new(),
        };

        for entry in self.batch_entries {
            registry.register_batch(entry)?;
        }

        for entry in self.ring_entries {
            registry.register_ring(entry)?;
        }

        info!(
            total = registry.total_count(),
            batch = registry.batch_kernel_ids().len(),
            ring = registry.ring_kernel_ids().len(),
            "Built kernel registry"
        );

        Ok(registry)
    }
}

/// Global kernel registry for auto-discovery.
///
/// This is used by the `#[gpu_kernel]` proc macro to automatically register kernels.
static GLOBAL_REGISTRY: std::sync::OnceLock<KernelRegistry> = std::sync::OnceLock::new();

/// Get or initialize the global registry.
pub fn global_registry() -> &'static KernelRegistry {
    GLOBAL_REGISTRY.get_or_init(KernelRegistry::new)
}

/// Initialize the global registry with a license.
///
/// Must be called before any kernel registration.
///
/// # Panics
///
/// Panics if the global registry has already been initialized.
pub fn init_global_registry(license: SharedLicenseValidator) -> &'static KernelRegistry {
    let registry = KernelRegistry::with_license(license);
    GLOBAL_REGISTRY
        .set(registry)
        .expect("Global registry already initialized");
    GLOBAL_REGISTRY.get().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::DevelopmentLicense;

    fn test_batch_entry() -> BatchKernelEntry {
        let metadata = KernelMetadata::batch("test-batch", Domain::Core);
        BatchKernelEntry::new(metadata, || {
            // Return a mock kernel
            panic!("Not implemented for tests")
        })
    }

    fn test_ring_entry() -> RingKernelEntry {
        let metadata = KernelMetadata::ring("test-ring", Domain::Core);
        RingKernelEntry::new(metadata, || {
            // Return a mock kernel
            panic!("Not implemented for tests")
        })
    }

    #[test]
    fn test_registry_creation() {
        let registry = KernelRegistry::new();
        assert_eq!(registry.total_count(), 0);
    }

    #[test]
    fn test_batch_registration() {
        let registry = KernelRegistry::new();
        let entry = test_batch_entry();

        registry.register_batch(entry).unwrap();
        assert_eq!(registry.total_count(), 1);
        assert!(registry.contains("test-batch"));
        assert!(registry.get_batch("test-batch").is_some());
    }

    #[test]
    fn test_ring_registration() {
        let registry = KernelRegistry::new();
        let entry = test_ring_entry();

        registry.register_ring(entry).unwrap();
        assert_eq!(registry.total_count(), 1);
        assert!(registry.contains("test-ring"));
        assert!(registry.get_ring("test-ring").is_some());
    }

    #[test]
    fn test_duplicate_registration() {
        let registry = KernelRegistry::new();
        let entry1 = test_batch_entry();
        let entry2 = test_batch_entry();

        registry.register_batch(entry1).unwrap();
        let result = registry.register_batch(entry2);
        assert!(result.is_err());
    }

    #[test]
    fn test_by_domain() {
        let registry = KernelRegistry::new();

        let core_entry = test_batch_entry();
        registry.register_batch(core_entry).unwrap();

        let graph_entry = BatchKernelEntry::new(
            KernelMetadata::batch("test-graph", Domain::GraphAnalytics),
            || panic!("Not implemented"),
        );
        registry.register_batch(graph_entry).unwrap();

        let core_kernels = registry.by_domain(Domain::Core);
        assert_eq!(core_kernels.len(), 1);

        let graph_kernels = registry.by_domain(Domain::GraphAnalytics);
        assert_eq!(graph_kernels.len(), 1);
    }

    #[test]
    fn test_stats() {
        let registry = KernelRegistry::new();

        registry.register_batch(test_batch_entry()).unwrap();
        registry.register_ring(test_ring_entry()).unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.batch_kernels, 1);
        assert_eq!(stats.ring_kernels, 1);
        assert_eq!(stats.by_domain.get(&Domain::Core), Some(&2));
    }

    #[test]
    fn test_unregister() {
        let registry = KernelRegistry::new();
        registry.register_batch(test_batch_entry()).unwrap();

        assert!(registry.contains("test-batch"));
        assert!(registry.unregister("test-batch"));
        assert!(!registry.contains("test-batch"));
        assert!(!registry.unregister("test-batch"));
    }

    #[test]
    fn test_with_license() {
        let license: SharedLicenseValidator = Arc::new(DevelopmentLicense);
        let registry = KernelRegistry::with_license(license);

        // Should succeed with dev license
        registry.register_batch(test_batch_entry()).unwrap();
        registry.register_ring(test_ring_entry()).unwrap();
    }

    #[test]
    fn test_builder() {
        let registry = KernelRegistryBuilder::new()
            .with_batch(test_batch_entry())
            .with_ring(test_ring_entry())
            .build()
            .unwrap();

        assert_eq!(registry.total_count(), 2);
    }

    #[test]
    fn test_metadata_only_registration() {
        let registry = KernelRegistry::new();
        let metadata = KernelMetadata::batch("test-metadata", Domain::GraphAnalytics);

        registry.register_metadata(metadata).unwrap();

        assert_eq!(registry.total_count(), 1);
        assert!(registry.contains("test-metadata"));
        assert!(registry.get("test-metadata").is_some());
        assert!(registry.get_metadata_only("test-metadata").is_some());
        assert!(registry.get_batch("test-metadata").is_none()); // Not a batch entry
    }

    #[test]
    fn test_metadata_only_duplicate() {
        let registry = KernelRegistry::new();
        let metadata1 = KernelMetadata::batch("test-dup", Domain::Core);
        let metadata2 = KernelMetadata::batch("test-dup", Domain::Core);

        registry.register_metadata(metadata1).unwrap();
        let result = registry.register_metadata(metadata2);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_only_conflict_with_batch() {
        let registry = KernelRegistry::new();
        registry.register_batch(test_batch_entry()).unwrap();

        // Try to register metadata with same ID
        let metadata = KernelMetadata::batch("test-batch", Domain::Core);
        let result = registry.register_metadata(metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_only_in_stats() {
        let registry = KernelRegistry::new();

        let batch_meta = KernelMetadata::batch("meta-batch", Domain::GraphAnalytics);
        let ring_meta = KernelMetadata::ring("meta-ring", Domain::GraphAnalytics);
        registry.register_metadata(batch_meta).unwrap();
        registry.register_metadata(ring_meta).unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.batch_kernels, 1);
        assert_eq!(stats.ring_kernels, 1);
        assert_eq!(stats.by_domain.get(&Domain::GraphAnalytics), Some(&2));
    }

    #[test]
    fn test_metadata_only_by_domain() {
        let registry = KernelRegistry::new();

        let graph_meta = KernelMetadata::batch("graph-kernel", Domain::GraphAnalytics);
        let ml_meta = KernelMetadata::batch("ml-kernel", Domain::StatisticalML);
        registry.register_metadata(graph_meta).unwrap();
        registry.register_metadata(ml_meta).unwrap();

        let graph_kernels = registry.by_domain(Domain::GraphAnalytics);
        assert_eq!(graph_kernels.len(), 1);
        assert_eq!(graph_kernels[0].id, "graph-kernel");

        let ml_kernels = registry.by_domain(Domain::StatisticalML);
        assert_eq!(ml_kernels.len(), 1);
        assert_eq!(ml_kernels[0].id, "ml-kernel");
    }

    #[test]
    fn test_unregister_metadata_only() {
        let registry = KernelRegistry::new();
        let metadata = KernelMetadata::batch("test-unreg", Domain::Core);

        registry.register_metadata(metadata).unwrap();
        assert!(registry.contains("test-unreg"));

        assert!(registry.unregister("test-unreg"));
        assert!(!registry.contains("test-unreg"));
    }
}
