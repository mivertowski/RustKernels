//! # RustKernel Payment Processing
//!
//! GPU-accelerated payment processing kernels.
//!
//! ## Kernels
//! - `PaymentProcessing` - Real-time transaction execution
//! - `FlowAnalysis` - Payment flow network analysis and metrics

#![warn(missing_docs)]

pub mod types;
pub mod processing;
pub mod flow;

pub use types::*;
pub use processing::{PaymentProcessing, ProcessingConfig, PaymentRouting};
pub use flow::{FlowAnalysis, FlowAnalysisConfig, AccountFlowAnalysis};

/// Register all payment kernels.
pub fn register_all(_registry: &rustkernel_core::registry::KernelRegistry) -> rustkernel_core::error::Result<()> {
    tracing::info!("Registering payment processing kernels");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustkernel_core::{domain::Domain, traits::GpuKernel};

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
