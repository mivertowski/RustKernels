//! # RustKernels
//!
//! GPU-accelerated kernel library for financial services, analytics, and compliance workloads.
//!
//! RustKernels is a Rust port of the DotCompute GPU kernel library, leveraging the
//! RustCompute (RingKernel) framework for GPU-native persistent actors.
//!
//! ## Features
//!
//! - **16 Domain Categories**: Graph analytics, ML, compliance, risk, temporal analysis, and more
//! - **173+ Kernels**: Comprehensive coverage of financial and analytical algorithms
//! - **Dual Execution Modes**:
//!   - **Batch**: CPU-orchestrated, 10-50μs overhead, for periodic heavy computation
//!   - **Ring**: GPU-persistent actor, 100-500ns latency, for high-frequency operations
//! - **Enterprise Licensing**: Domain-based licensing and feature gating
//! - **Multi-Backend**: CUDA, WebGPU, and CPU backends via RustCompute
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rustkernel::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a kernel registry
//!     let registry = KernelRegistry::new();
//!
//!     // Get runtime with best available backend
//!     let runtime = RingKernel::builder()
//!         .backend(Backend::Auto)
//!         .build()
//!         .await?;
//!
//!     // Launch a specific kernel
//!     let pagerank = runtime.launch("graph/pagerank", LaunchOptions::default()).await?;
//!
//!     // Use it
//!     pagerank.send(PageRankRequest { node_id: 42, operation: PageRankOp::Query }).await?;
//!     let response: PageRankResponse = pagerank.receive().await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Domain Organization
//!
//! Kernels are organized into domains representing different business/analytical areas:
//!
//! ### Priority 1 (High Value)
//! - **GraphAnalytics**: Centrality, community detection, motifs, similarity
//! - **StatisticalML**: Clustering, anomaly detection, regression
//! - **Compliance**: AML, KYC, sanctions screening
//! - **TemporalAnalysis**: Forecasting, change detection, decomposition
//! - **RiskAnalytics**: Credit risk, VaR, portfolio risk
//!
//! ### Priority 2 (Medium)
//! - **Banking**: Fraud detection
//! - **BehavioralAnalytics**: Profiling, forensics
//! - **OrderMatching**: HFT order book
//! - **ProcessIntelligence**: Process mining
//! - **Clearing**: Settlement, netting
//!
//! ### Priority 3 (Lower)
//! - **TreasuryManagement**: Cash flow, hedging
//! - **Accounting**: Chart of accounts, reconciliation
//! - **PaymentProcessing**: Transaction execution
//! - **FinancialAudit**: Feature extraction
//!
//! ## Feature Flags
//!
//! Enable domain crates via Cargo features:
//!
//! ```toml
//! [dependencies]
//! rustkernel = { version = "0.1", features = ["graph", "ml", "risk"] }
//! ```
//!
//! Available features:
//! - `default`: graph, ml, compliance, temporal, risk
//! - `full`: All domains
//! - Individual: `graph`, `ml`, `compliance`, `temporal`, `risk`, `banking`, etc.

#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export core crate
pub use rustkernel_core as core;

// Re-export derive macros
pub use rustkernel_derive::{gpu_kernel, kernel_state, KernelMessage};

// Re-export ringkernel for direct access
pub use ringkernel;
pub use ringkernel_core;

// Domain re-exports (conditional on features)
#[cfg(feature = "graph")]
pub use rustkernel_graph as graph;

#[cfg(feature = "ml")]
pub use rustkernel_ml as ml;

#[cfg(feature = "compliance")]
pub use rustkernel_compliance as compliance;

#[cfg(feature = "temporal")]
pub use rustkernel_temporal as temporal;

#[cfg(feature = "risk")]
pub use rustkernel_risk as risk;

#[cfg(feature = "banking")]
pub use rustkernel_banking as banking;

#[cfg(feature = "behavioral")]
pub use rustkernel_behavioral as behavioral;

#[cfg(feature = "orderbook")]
pub use rustkernel_orderbook as orderbook;

#[cfg(feature = "procint")]
pub use rustkernel_procint as procint;

#[cfg(feature = "clearing")]
pub use rustkernel_clearing as clearing;

#[cfg(feature = "treasury")]
pub use rustkernel_treasury as treasury;

#[cfg(feature = "accounting")]
pub use rustkernel_accounting as accounting;

#[cfg(feature = "payments")]
pub use rustkernel_payments as payments;

#[cfg(feature = "audit")]
pub use rustkernel_audit as audit;

/// Prelude module for convenient imports.
///
/// Import everything you need with:
/// ```rust,ignore
/// use rustkernel::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use rustkernel_core::prelude::*;

    // Derive macros
    pub use rustkernel_derive::{gpu_kernel, kernel_state, KernelMessage};

    // RingKernel types
    pub use ringkernel_core::{
        HlcTimestamp, KernelHandle, KernelId, KernelState, LaunchOptions, MessageId, RingContext,
        RingMessage,
    };
}

/// Version information.
pub mod version {
    /// Crate version.
    pub const VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Minimum supported RustCompute version.
    pub const MIN_RINGKERNEL_VERSION: &str = "0.1.0";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        // Verify core types are accessible
        let _domain = Domain::GraphAnalytics;
        let _mode = KernelMode::Ring;
    }

    #[test]
    fn test_version() {
        assert!(!version::VERSION.is_empty());
    }
}
