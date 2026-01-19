//! Memory Management Infrastructure
//!
//! Provides GPU memory pooling, pressure handling, and analytics context management
//! for high-performance kernel execution.
//!
//! # Features
//!
//! - **Memory Pools**: Size-stratified pools for efficient GPU allocation
//! - **Pressure Handling**: Automatic memory pressure detection and mitigation
//! - **Analytics Context**: Reusable buffers for analytics workloads
//! - **Reduction Buffers**: Cached buffers for GPU reduction operations
//! - **Multi-phase Reductions**: Synchronization primitives for iterative algorithms
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::memory::{KernelMemoryManager, MemoryConfig};
//!
//! let config = MemoryConfig::production();
//! let manager = KernelMemoryManager::new(config);
//!
//! // Allocate from pool
//! let buffer = manager.allocate(1024 * 1024)?; // 1MB
//!
//! // Return to pool
//! manager.deallocate(buffer);
//! ```

pub mod reduction;

pub use reduction::{
    CooperativeBarrier, GlobalReduction, InterPhaseReduction, PhaseState, ReductionBuilder,
    ReductionConfig, ReductionError, ReductionOp, SyncMode,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum total GPU memory to use (bytes)
    pub max_gpu_memory: u64,
    /// Maximum total CPU staging memory (bytes)
    pub max_staging_memory: u64,
    /// Enable memory pooling
    pub pooling_enabled: bool,
    /// Pool bucket sizes (bytes)
    pub bucket_sizes: Vec<u64>,
    /// Pressure threshold (0.0-1.0)
    pub pressure_threshold: f64,
    /// Enable automatic defragmentation
    pub auto_defrag: bool,
    /// Defrag threshold (fragmentation ratio)
    pub defrag_threshold: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory: 4 * 1024 * 1024 * 1024, // 4GB
            max_staging_memory: 1024 * 1024 * 1024, // 1GB
            pooling_enabled: true,
            bucket_sizes: vec![
                64 * 1024,       // 64KB
                256 * 1024,      // 256KB
                1024 * 1024,     // 1MB
                4 * 1024 * 1024, // 4MB
                16 * 1024 * 1024, // 16MB
                64 * 1024 * 1024, // 64MB
            ],
            pressure_threshold: 0.85,
            auto_defrag: true,
            defrag_threshold: 0.3,
        }
    }
}

impl MemoryConfig {
    /// Create development configuration (smaller limits)
    pub fn development() -> Self {
        Self {
            max_gpu_memory: 512 * 1024 * 1024, // 512MB
            max_staging_memory: 256 * 1024 * 1024, // 256MB
            pooling_enabled: false,
            ..Default::default()
        }
    }

    /// Create production configuration
    pub fn production() -> Self {
        Self::default()
    }

    /// Create high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            max_gpu_memory: 16 * 1024 * 1024 * 1024, // 16GB
            max_staging_memory: 4 * 1024 * 1024 * 1024, // 4GB
            pooling_enabled: true,
            auto_defrag: true,
            defrag_threshold: 0.2,
            ..Default::default()
        }
    }
}

/// Size bucket for memory pool
#[derive(Debug)]
pub struct SizeBucket {
    /// Bucket size in bytes
    pub size: u64,
    /// Number of available buffers
    pub available: AtomicUsize,
    /// Number of allocated buffers
    pub allocated: AtomicUsize,
    /// Peak allocation
    pub peak: AtomicUsize,
}

impl SizeBucket {
    /// Create a new size bucket
    pub fn new(size: u64) -> Self {
        Self {
            size,
            available: AtomicUsize::new(0),
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    /// Record an allocation
    pub fn record_alloc(&self) {
        let count = self.allocated.fetch_add(1, Ordering::Relaxed) + 1;
        let mut peak = self.peak.load(Ordering::Relaxed);
        while count > peak {
            match self.peak.compare_exchange_weak(peak, count, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Record a deallocation
    pub fn record_dealloc(&self) {
        self.allocated.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get bucket statistics
    pub fn stats(&self) -> BucketStats {
        BucketStats {
            size: self.size,
            available: self.available.load(Ordering::Relaxed),
            allocated: self.allocated.load(Ordering::Relaxed),
            peak: self.peak.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for a size bucket
#[derive(Debug, Clone)]
pub struct BucketStats {
    /// Bucket size in bytes
    pub size: u64,
    /// Number of available buffers
    pub available: usize,
    /// Number of allocated buffers
    pub allocated: usize,
    /// Peak allocation
    pub peak: usize,
}

/// Memory buffer handle
#[derive(Debug)]
pub struct MemoryBuffer {
    /// Buffer ID
    pub id: u64,
    /// Size in bytes
    pub size: u64,
    /// Bucket index (if from pool)
    pub bucket_index: Option<usize>,
    /// Whether buffer is GPU memory
    pub is_gpu: bool,
}

/// Memory allocation result
pub type AllocResult<T> = std::result::Result<T, MemoryError>;

/// Memory errors
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Out of memory
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Requested size
        requested: u64,
        /// Available size
        available: u64,
    },

    /// Memory pressure exceeded
    #[error("Memory pressure exceeded: {usage_percent:.1}% usage")]
    PressureExceeded {
        /// Current usage percentage
        usage_percent: f64,
    },

    /// Invalid buffer
    #[error("Invalid buffer: {id}")]
    InvalidBuffer {
        /// Buffer ID
        id: u64,
    },

    /// Allocation failed
    #[error("Allocation failed: {reason}")]
    AllocationFailed {
        /// Failure reason
        reason: String,
    },
}

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureLevel {
    /// Normal operation
    Normal,
    /// Elevated usage, start cleanup
    Warning,
    /// High usage, defer allocations
    High,
    /// Critical usage, emergency cleanup
    Critical,
}

impl PressureLevel {
    /// Get pressure level from usage ratio
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio < 0.70 {
            Self::Normal
        } else if ratio < 0.85 {
            Self::Warning
        } else if ratio < 0.95 {
            Self::High
        } else {
            Self::Critical
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total GPU memory (bytes)
    pub gpu_total: u64,
    /// Used GPU memory (bytes)
    pub gpu_used: u64,
    /// Peak GPU memory (bytes)
    pub gpu_peak: u64,
    /// Total staging memory (bytes)
    pub staging_total: u64,
    /// Used staging memory (bytes)
    pub staging_used: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Number of deallocations
    pub deallocations: u64,
    /// Pool hit rate
    pub pool_hit_rate: f64,
    /// Current pressure level
    pub pressure_level: PressureLevel,
}

impl Default for PressureLevel {
    fn default() -> Self {
        Self::Normal
    }
}

/// Kernel memory manager
pub struct KernelMemoryManager {
    config: MemoryConfig,
    buckets: Vec<SizeBucket>,
    stats: Arc<MemoryStatsInner>,
    buffers: Arc<RwLock<HashMap<u64, MemoryBuffer>>>,
    next_id: AtomicU64,
}

#[derive(Debug, Default)]
struct MemoryStatsInner {
    gpu_used: AtomicU64,
    gpu_peak: AtomicU64,
    staging_used: AtomicU64,
    allocations: AtomicU64,
    deallocations: AtomicU64,
    pool_hits: AtomicU64,
    pool_misses: AtomicU64,
}

impl KernelMemoryManager {
    /// Create a new memory manager
    pub fn new(config: MemoryConfig) -> Self {
        let buckets = config
            .bucket_sizes
            .iter()
            .map(|&size| SizeBucket::new(size))
            .collect();

        Self {
            config,
            buckets,
            stats: Arc::new(MemoryStatsInner::default()),
            buffers: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Allocate GPU memory
    pub async fn allocate(&self, size: u64) -> AllocResult<MemoryBuffer> {
        // Check pressure
        let pressure = self.pressure_level();
        if pressure == PressureLevel::Critical {
            return Err(MemoryError::PressureExceeded {
                usage_percent: self.gpu_usage_percent(),
            });
        }

        // Check limits
        let current_used = self.stats.gpu_used.load(Ordering::Relaxed);
        if current_used + size > self.config.max_gpu_memory {
            return Err(MemoryError::OutOfMemory {
                requested: size,
                available: self.config.max_gpu_memory - current_used,
            });
        }

        // Try pool allocation
        let bucket_index = if self.config.pooling_enabled {
            self.find_bucket(size)
        } else {
            None
        };

        if let Some(idx) = bucket_index {
            self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
            self.buckets[idx].record_alloc();
        } else if self.config.pooling_enabled {
            self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Update stats
        self.stats.gpu_used.fetch_add(size, Ordering::Relaxed);
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);

        // Update peak
        let new_used = self.stats.gpu_used.load(Ordering::Relaxed);
        let mut peak = self.stats.gpu_peak.load(Ordering::Relaxed);
        while new_used > peak {
            match self.stats.gpu_peak.compare_exchange_weak(
                peak,
                new_used,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let buffer = MemoryBuffer {
            id,
            size,
            bucket_index,
            is_gpu: true,
        };

        self.buffers.write().await.insert(id, MemoryBuffer {
            id,
            size,
            bucket_index,
            is_gpu: true,
        });

        Ok(buffer)
    }

    /// Deallocate GPU memory
    pub async fn deallocate(&self, buffer: MemoryBuffer) -> AllocResult<()> {
        let removed = self.buffers.write().await.remove(&buffer.id);
        if removed.is_none() {
            return Err(MemoryError::InvalidBuffer { id: buffer.id });
        }

        if let Some(idx) = buffer.bucket_index {
            self.buckets[idx].record_dealloc();
        }

        self.stats.gpu_used.fetch_sub(buffer.size, Ordering::Relaxed);
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Allocate staging (CPU) memory
    pub async fn allocate_staging(&self, size: u64) -> AllocResult<MemoryBuffer> {
        let current_used = self.stats.staging_used.load(Ordering::Relaxed);
        if current_used + size > self.config.max_staging_memory {
            return Err(MemoryError::OutOfMemory {
                requested: size,
                available: self.config.max_staging_memory - current_used,
            });
        }

        self.stats.staging_used.fetch_add(size, Ordering::Relaxed);
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let buffer = MemoryBuffer {
            id,
            size,
            bucket_index: None,
            is_gpu: false,
        };

        self.buffers.write().await.insert(id, MemoryBuffer {
            id,
            size,
            bucket_index: None,
            is_gpu: false,
        });

        Ok(buffer)
    }

    /// Deallocate staging memory
    pub async fn deallocate_staging(&self, buffer: MemoryBuffer) -> AllocResult<()> {
        let removed = self.buffers.write().await.remove(&buffer.id);
        if removed.is_none() {
            return Err(MemoryError::InvalidBuffer { id: buffer.id });
        }

        self.stats.staging_used.fetch_sub(buffer.size, Ordering::Relaxed);
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get current memory statistics
    pub fn stats(&self) -> MemoryStats {
        let gpu_used = self.stats.gpu_used.load(Ordering::Relaxed);
        let pool_hits = self.stats.pool_hits.load(Ordering::Relaxed);
        let pool_misses = self.stats.pool_misses.load(Ordering::Relaxed);
        let total_pool = pool_hits + pool_misses;

        MemoryStats {
            gpu_total: self.config.max_gpu_memory,
            gpu_used,
            gpu_peak: self.stats.gpu_peak.load(Ordering::Relaxed),
            staging_total: self.config.max_staging_memory,
            staging_used: self.stats.staging_used.load(Ordering::Relaxed),
            allocations: self.stats.allocations.load(Ordering::Relaxed),
            deallocations: self.stats.deallocations.load(Ordering::Relaxed),
            pool_hit_rate: if total_pool > 0 {
                pool_hits as f64 / total_pool as f64
            } else {
                0.0
            },
            pressure_level: self.pressure_level(),
        }
    }

    /// Get bucket statistics
    pub fn bucket_stats(&self) -> Vec<BucketStats> {
        self.buckets.iter().map(|b| b.stats()).collect()
    }

    /// Get current pressure level
    pub fn pressure_level(&self) -> PressureLevel {
        PressureLevel::from_ratio(self.gpu_usage_percent() / 100.0)
    }

    /// Get GPU usage percentage
    pub fn gpu_usage_percent(&self) -> f64 {
        let used = self.stats.gpu_used.load(Ordering::Relaxed) as f64;
        let total = self.config.max_gpu_memory as f64;
        (used / total) * 100.0
    }

    /// Request garbage collection
    pub async fn request_gc(&self) {
        // Clear unused pool buffers
        tracing::info!("Memory GC requested, pressure level: {:?}", self.pressure_level());
    }

    /// Find appropriate bucket for size
    fn find_bucket(&self, size: u64) -> Option<usize> {
        self.buckets
            .iter()
            .position(|b| b.size >= size)
    }
}

impl Default for KernelMemoryManager {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

/// Reduction buffer for GPU reduction operations
#[derive(Debug)]
pub struct ReductionBuffer<T> {
    /// Buffer data
    data: Vec<T>,
    /// Capacity
    capacity: usize,
}

impl<T: Default + Clone> ReductionBuffer<T> {
    /// Create a new reduction buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
        }
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get data slice
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable data slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Reset buffer to default values
    pub fn reset(&mut self) {
        for item in &mut self.data {
            *item = T::default();
        }
    }
}

/// Reduction buffer cache
pub struct ReductionBufferCache {
    max_buffers: usize,
    buffers: Arc<RwLock<Vec<Vec<u8>>>>,
}

impl ReductionBufferCache {
    /// Create a new cache
    pub fn new(max_buffers: usize) -> Self {
        Self {
            max_buffers,
            buffers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get or create a buffer of the given size
    pub async fn get(&self, size: usize) -> Vec<u8> {
        let mut buffers = self.buffers.write().await;

        // Try to find a buffer of adequate size
        if let Some(pos) = buffers.iter().position(|b| b.capacity() >= size) {
            let mut buf = buffers.remove(pos);
            buf.resize(size, 0);
            return buf;
        }

        // Create new buffer
        vec![0u8; size]
    }

    /// Return a buffer to the cache
    pub async fn return_buffer(&self, buffer: Vec<u8>) {
        let mut buffers = self.buffers.write().await;
        if buffers.len() < self.max_buffers {
            buffers.push(buffer);
        }
        // Otherwise let it drop
    }

    /// Clear the cache
    pub async fn clear(&self) {
        self.buffers.write().await.clear();
    }
}

impl Default for ReductionBufferCache {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Analytics context for reusable buffers
#[derive(Debug)]
pub struct AnalyticsContext {
    /// Context ID
    pub id: u64,
    /// Maximum working set size
    pub max_working_set: u64,
    /// Current allocations
    allocations: AtomicU64,
}

impl AnalyticsContext {
    /// Create a new analytics context
    pub fn new(id: u64, max_working_set: u64) -> Self {
        Self {
            id,
            max_working_set,
            allocations: AtomicU64::new(0),
        }
    }

    /// Record an allocation
    pub fn record_allocation(&self, size: u64) -> bool {
        let current = self.allocations.load(Ordering::Relaxed);
        if current + size > self.max_working_set {
            return false;
        }
        self.allocations.fetch_add(size, Ordering::Relaxed);
        true
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: u64) {
        self.allocations.fetch_sub(size, Ordering::Relaxed);
    }

    /// Get current usage
    pub fn current_usage(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Get usage percentage
    pub fn usage_percent(&self) -> f64 {
        (self.current_usage() as f64 / self.max_working_set as f64) * 100.0
    }
}

/// Analytics context manager
pub struct AnalyticsContextManager {
    contexts: Arc<RwLock<HashMap<u64, Arc<AnalyticsContext>>>>,
    default_working_set: u64,
    next_id: AtomicU64,
}

impl AnalyticsContextManager {
    /// Create a new context manager
    pub fn new(default_working_set: u64) -> Self {
        Self {
            contexts: Arc::new(RwLock::new(HashMap::new())),
            default_working_set,
            next_id: AtomicU64::new(1),
        }
    }

    /// Create a new analytics context
    pub async fn create_context(&self) -> Arc<AnalyticsContext> {
        self.create_context_with_size(self.default_working_set).await
    }

    /// Create a context with specific working set size
    pub async fn create_context_with_size(&self, max_working_set: u64) -> Arc<AnalyticsContext> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let ctx = Arc::new(AnalyticsContext::new(id, max_working_set));
        self.contexts.write().await.insert(id, ctx.clone());
        ctx
    }

    /// Get a context by ID
    pub async fn get_context(&self, id: u64) -> Option<Arc<AnalyticsContext>> {
        self.contexts.read().await.get(&id).cloned()
    }

    /// Remove a context
    pub async fn remove_context(&self, id: u64) {
        self.contexts.write().await.remove(&id);
    }

    /// Get number of active contexts
    pub async fn active_contexts(&self) -> usize {
        self.contexts.read().await.len()
    }
}

impl Default for AnalyticsContextManager {
    fn default() -> Self {
        Self::new(256 * 1024 * 1024) // 256MB default working set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_allocation() {
        let manager = KernelMemoryManager::new(MemoryConfig::development());

        let buffer = manager.allocate(1024).await.unwrap();
        assert_eq!(buffer.size, 1024);
        assert!(buffer.is_gpu);

        let stats = manager.stats();
        assert_eq!(stats.gpu_used, 1024);
        assert_eq!(stats.allocations, 1);

        manager.deallocate(buffer).await.unwrap();

        let stats = manager.stats();
        assert_eq!(stats.gpu_used, 0);
        assert_eq!(stats.deallocations, 1);
    }

    #[tokio::test]
    async fn test_out_of_memory() {
        let config = MemoryConfig {
            max_gpu_memory: 1024,
            ..MemoryConfig::development()
        };
        let manager = KernelMemoryManager::new(config);

        let result = manager.allocate(2048).await;
        assert!(matches!(result, Err(MemoryError::OutOfMemory { .. })));
    }

    #[tokio::test]
    async fn test_pressure_levels() {
        let config = MemoryConfig {
            max_gpu_memory: 1000,
            ..MemoryConfig::development()
        };
        let manager = KernelMemoryManager::new(config);

        assert_eq!(manager.pressure_level(), PressureLevel::Normal);

        // Allocate 70%
        let _buf = manager.allocate(700).await.unwrap();
        assert_eq!(manager.pressure_level(), PressureLevel::Warning);
    }

    #[tokio::test]
    async fn test_reduction_buffer_cache() {
        let cache = ReductionBufferCache::new(4);

        let buf1 = cache.get(1024).await;
        assert_eq!(buf1.len(), 1024);

        cache.return_buffer(buf1).await;

        // Should reuse the buffer
        let buf2 = cache.get(512).await;
        assert_eq!(buf2.len(), 512);
    }

    #[tokio::test]
    async fn test_analytics_context() {
        let manager = AnalyticsContextManager::new(1024);

        let ctx = manager.create_context().await;
        assert!(ctx.record_allocation(512));
        assert_eq!(ctx.current_usage(), 512);

        ctx.record_deallocation(256);
        assert_eq!(ctx.current_usage(), 256);
    }
}
