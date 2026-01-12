//! # RustKernel Payment Processing
//!
//! GPU-accelerated payment processing kernels.
//!
//! ## Kernels
//! - `PaymentProcessing` - Real-time transaction execution
//! - `FlowAnalysis` - Payment flow network analysis and metrics

#![warn(missing_docs)]

pub mod flow;
pub mod processing;
pub mod types;

pub use flow::{AccountFlowAnalysis, FlowAnalysis, FlowAnalysisConfig};
pub use processing::{PaymentProcessing, PaymentRouting, ProcessingConfig};
pub use types::*;

/// Register all payment kernels.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    use rustkernel_core::traits::GpuKernel;

    tracing::info!("Registering payment processing kernels");

    // Processing kernel (1)
    registry.register_metadata(processing::PaymentProcessing::new().metadata().clone())?;

    // Flow analysis kernel (1)
    registry.register_metadata(flow::FlowAnalysis::new().metadata().clone())?;

    tracing::info!("Registered 2 payment processing kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::{domain::Domain, registry::KernelRegistry, traits::GpuKernel};

    #[test]
    fn test_register_all() {
        let registry = KernelRegistry::new();
        register_all(&registry).expect("Failed to register payments kernels");
        assert_eq!(registry.total_count(), 2);
    }

    #[test]
    fn test_payment_processing_metadata() {
        let kernel = PaymentProcessing::new();
        let metadata = kernel.metadata();
        assert!(metadata.id.contains("processing"));
        assert_eq!(metadata.domain, Domain::PaymentProcessing);
    }

    #[test]
    fn test_flow_analysis_metadata() {
        let kernel = FlowAnalysis::new();
        let metadata = kernel.metadata();
        assert!(metadata.id.contains("flow"));
        assert_eq!(metadata.domain, Domain::PaymentProcessing);
    }
}
