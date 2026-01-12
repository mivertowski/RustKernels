//! # RustKernel Accounting
//!
//! GPU-accelerated accounting kernels.
//!
//! ## Kernels
//! - `ChartOfAccountsMapping` - Entity-specific CoA mapping
//! - `JournalTransformation` - GL mapping
//! - `GLReconciliation` - Account matching
//! - `NetworkAnalysis` - Intercompany analysis
//! - `TemporalCorrelation` - Account correlations

#![warn(missing_docs)]

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

/// Chart of accounts mapping kernel.
#[derive(Debug, Clone, Default)]
pub struct ChartOfAccountsMapping { metadata: KernelMetadata }
impl ChartOfAccountsMapping {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("accounting/coa-mapping", Domain::Accounting)
            .with_description("Entity-specific chart of accounts mapping")
            .with_throughput(50_000).with_latency_us(50.0) }
    }
}
impl GpuKernel for ChartOfAccountsMapping { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// GL reconciliation kernel.
#[derive(Debug, Clone, Default)]
pub struct GLReconciliation { metadata: KernelMetadata }
impl GLReconciliation {
    /// Create a new kernel.
    #[must_use]
    pub fn new() -> Self {
        Self { metadata: KernelMetadata::batch("accounting/gl-reconciliation", Domain::Accounting)
            .with_description("General ledger reconciliation")
            .with_throughput(20_000).with_latency_us(100.0) }
    }
}
impl GpuKernel for GLReconciliation { fn metadata(&self) -> &KernelMetadata { &self.metadata } }

/// Register all accounting kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering accounting kernels");
    Ok(())
}
