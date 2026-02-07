//! # RustKernel Process Intelligence
//!
//! GPU-accelerated process mining and conformance checking.
//!
//! ## Kernels
//! - `DFGConstruction` - Directly-follows graph construction
//! - `PartialOrderAnalysis` - Concurrency detection
//! - `ConformanceChecking` - Multi-model conformance (DFG/Petri/BPMN)
//! - `OCPMPatternMatching` - Object-centric process mining
//! - `NextActivityPrediction` - Markov/N-gram next activity prediction
//! - `EventLogImputation` - Event log quality detection and repair
//! - `DigitalTwin` - Process simulation for what-if analysis
//!
//! ## Features
//! - Directly-follows graph construction from event logs
//! - Partial order analysis for concurrency detection
//! - Conformance checking against DFG and Petri net models
//! - Object-centric process mining for multi-object workflows
//! - Next activity prediction using Markov chains and N-grams
//! - Event log imputation for missing events and timestamp repair

#![warn(missing_docs)]

pub mod conformance;
pub mod dfg;
pub mod imputation;
pub mod ocpm;
pub mod partial_order;
pub mod prediction;
pub mod simulation;
pub mod types;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::conformance::*;
    pub use crate::dfg::*;
    pub use crate::imputation::*;
    pub use crate::ocpm::*;
    pub use crate::partial_order::*;
    pub use crate::prediction::*;
    pub use crate::simulation::*;
    pub use crate::types::*;
}

// Re-export main kernels
pub use conformance::ConformanceChecking;
pub use dfg::DFGConstruction;
pub use imputation::EventLogImputation;
pub use ocpm::OCPMPatternMatching;
pub use partial_order::PartialOrderAnalysis;
pub use prediction::NextActivityPrediction;
pub use simulation::DigitalTwin;

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
    tracing::info!("Registering process intelligence kernels");

    // DFG kernel (1) - Batch
    registry.register_batch_metadata_from(dfg::DFGConstruction::new)?;

    // Partial order kernel (1) - Batch
    registry.register_batch_metadata_from(partial_order::PartialOrderAnalysis::new)?;

    // Conformance kernel (1) - Ring
    registry.register_ring_metadata_from(conformance::ConformanceChecking::new)?;

    // OCPM kernel (1) - Batch
    registry.register_batch_metadata_from(ocpm::OCPMPatternMatching::new)?;

    // Prediction kernel (1) - Batch
    registry.register_batch_metadata_from(prediction::NextActivityPrediction::new)?;

    // Imputation kernel (1) - Batch
    registry.register_batch_metadata_from(imputation::EventLogImputation::new)?;

    // Simulation kernel (1) - Batch
    registry.register_batch_metadata_from(simulation::DigitalTwin::new)?;

    tracing::info!("Registered 7 process intelligence kernels");
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
        assert_eq!(registry.total_count(), 7);
    }
}
