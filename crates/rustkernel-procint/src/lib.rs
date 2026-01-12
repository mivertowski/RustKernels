//! # RustKernel Process Intelligence
//!
//! GPU-accelerated process mining and conformance checking.
//!
//! ## Kernels
//! - `DFGConstruction` - Directly-follows graph construction
//! - `PartialOrderAnalysis` - Concurrency detection
//! - `ConformanceChecking` - Multi-model conformance (DFG/Petri/BPMN)
//! - `OCPMPatternMatching` - Object-centric process mining
//!
//! ## Features
//! - Directly-follows graph construction from event logs
//! - Partial order analysis for concurrency detection
//! - Conformance checking against DFG and Petri net models
//! - Object-centric process mining for multi-object workflows

#![warn(missing_docs)]

pub mod conformance;
pub mod dfg;
pub mod ocpm;
pub mod partial_order;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::conformance::*;
    pub use crate::dfg::*;
    pub use crate::ocpm::*;
    pub use crate::partial_order::*;
    pub use crate::types::*;
}

// Re-export main kernels
pub use conformance::ConformanceChecking;
pub use dfg::DFGConstruction;
pub use ocpm::OCPMPatternMatching;
pub use partial_order::PartialOrderAnalysis;

// Re-export key types
pub use types::{
    AlignmentStep, Arc, ConformanceResult, ConformanceStats, DFGEdge, DFGResult, Deviation,
    DeviationType, DirectlyFollowsGraph, EventLog, OCPMEvent, OCPMEventLog, OCPMObject,
    OCPMPatternResult, PartialOrderResult, PetriNet, Place, ProcessEvent, Trace, Transition,
};

/// Register all process intelligence kernels with a registry.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering process intelligence kernels");

    // DFG kernel (1)
    registry.register_metadata(dfg::DFGConstruction::new().metadata().clone())?;

    // Partial order kernel (1)
    registry.register_metadata(
        partial_order::PartialOrderAnalysis::new()
            .metadata()
            .clone(),
    )?;

    // Conformance kernel (1)
    registry.register_metadata(conformance::ConformanceChecking::new().metadata().clone())?;

    // OCPM kernel (1)
    registry.register_metadata(ocpm::OCPMPatternMatching::new().metadata().clone())?;

    tracing::info!("Registered 4 process intelligence kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::registry::KernelRegistry;

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register procint kernels");
        assert_eq!(registry.total_count(), 4);
    }
}
