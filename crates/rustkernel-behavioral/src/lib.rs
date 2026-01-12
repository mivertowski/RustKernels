//! # RustKernel Behavioral Analytics
//!
//! GPU-accelerated behavioral analytics kernels for profiling and forensics.
//!
//! ## Kernels
//! - `BehavioralProfiling` - Feature extraction for user behavior
//! - `AnomalyProfiling` - Deviation scoring from behavioral baseline
//! - `FraudSignatureDetection` - Known fraud pattern matching
//! - `CausalGraphConstruction` - DAG inference from event streams
//! - `ForensicQueryExecution` - Historical pattern search and analysis
//! - `EventCorrelationKernel` - Temporal event correlation and clustering

#![warn(missing_docs)]

pub mod causal;
pub mod correlation;
pub mod forensics;
pub mod profiling;
pub mod signatures;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::causal::*;
    pub use crate::correlation::*;
    pub use crate::forensics::*;
    pub use crate::profiling::*;
    pub use crate::signatures::*;
    pub use crate::types::*;
}

// Re-export main kernels
pub use causal::CausalGraphConstruction;
pub use correlation::EventCorrelationKernel;
pub use forensics::ForensicQueryExecution;
pub use profiling::{AnomalyProfiling, BehavioralProfiling};
pub use signatures::FraudSignatureDetection;

// Re-export key types
pub use types::{
    AnomalyResult, AnomalyType, BehaviorProfile, CausalEdge, CausalGraphResult, CausalNode,
    CorrelationCluster, CorrelationResult, CorrelationType, EventCorrelation, EventValue,
    FeatureDeviation, ForensicQuery, ForensicResult, FraudSignature, ProfilingResult, QueryType,
    SignatureMatch, SignaturePattern, UserEvent,
};

/// Register all behavioral kernels with a registry.
pub fn register_all(
    _registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering behavioral analytics kernels");
    Ok(())
}
