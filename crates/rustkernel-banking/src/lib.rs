//! # RustKernel Banking
//!
//! GPU-accelerated banking kernels for fraud detection.
//!
//! ## Kernels
//! - `FraudPatternMatch` - Multi-pattern fraud detection combining:
//!   - Aho-Corasick pattern matching
//!   - Rapid split (structuring) detection
//!   - Circular flow detection
//!   - Velocity and amount anomalies
//!   - Geographic anomaly (impossible travel)
//!   - Mule account detection

#![warn(missing_docs)]

pub mod fraud;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::fraud::*;
    pub use crate::types::*;
}

// Re-export main kernel
pub use fraud::FraudPatternMatch;

// Re-export key types
pub use types::{
    AccountProfile, BankTransaction, Channel, FraudDetectionResult, FraudPattern, FraudPatternType,
    PatternMatch, PatternParams, RecommendedAction, RiskLevel, TransactionType,
};

/// Register all banking kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering banking kernels");

    // Fraud detection kernel (1) - Ring
    registry.register_ring_metadata_from(fraud::FraudPatternMatch::new)?;

    tracing::info!("Registered 1 banking kernel");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register banking kernels");
        assert_eq!(registry.total_count(), 1);
    }
}
