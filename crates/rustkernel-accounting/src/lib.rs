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

pub mod types;
pub mod coa_mapping;
pub mod journal;
pub mod reconciliation;
pub mod network;
pub mod temporal;

pub use coa_mapping::ChartOfAccountsMapping;
pub use journal::JournalTransformation;
pub use reconciliation::GLReconciliation;
pub use network::NetworkAnalysis;
pub use temporal::TemporalCorrelation;

/// Register all accounting kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering accounting kernels");
    Ok(())
}
