//! # RustKernel Compliance
//!
//! GPU-accelerated compliance kernels for AML, KYC, sanctions screening, and transaction monitoring.
//!
//! ## Kernels
//!
//! ### AML (4 kernels)
//! - `CircularFlowRatio` - SCC detection for circular transactions
//! - `ReciprocityFlowRatio` - Mutual transaction detection
//! - `RapidMovement` - Velocity analysis for structuring
//! - `AMLPatternDetection` - Multi-pattern FSM detection
//!
//! ### KYC (2 kernels)
//! - `KYCScoring` - Risk factor aggregation
//! - `EntityResolution` - Fuzzy entity matching
//!
//! ### Sanctions (2 kernels)
//! - `SanctionsScreening` - OFAC/UN/EU list matching
//! - `PEPScreening` - Politically exposed persons
//!
//! ### Monitoring (1 kernel)
//! - `TransactionMonitoring` - Real-time threshold alerts

#![warn(missing_docs)]

pub mod aml;
pub mod kyc;
pub mod monitoring;
pub mod sanctions;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::aml::*;
    pub use crate::kyc::*;
    pub use crate::monitoring::*;
    pub use crate::sanctions::*;
    pub use crate::types::*;
}

// Re-export main types for convenience
pub use aml::{AMLPatternDetection, CircularFlowRatio, RapidMovement, ReciprocityFlowRatio};
pub use kyc::{EntityResolution, KYCScoring};
pub use monitoring::TransactionMonitoring;
pub use sanctions::{PEPScreening, SanctionsScreening};

/// Register all compliance kernels with a registry.
pub fn register_all(
    _registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering compliance kernels");
    Ok(())
}
