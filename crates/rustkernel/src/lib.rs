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

/// Kernel catalog providing overview of all available kernels.
pub mod catalog {
    use rustkernel_core::domain::Domain;

    /// Domain information.
    #[derive(Debug, Clone)]
    pub struct DomainInfo {
        /// Domain enum value.
        pub domain: Domain,
        /// Human-readable name.
        pub name: &'static str,
        /// Description.
        pub description: &'static str,
        /// Number of kernels.
        pub kernel_count: usize,
        /// Feature flag to enable.
        pub feature: &'static str,
    }

    /// Get all domain information.
    pub fn domains() -> Vec<DomainInfo> {
        vec![
            DomainInfo {
                domain: Domain::GraphAnalytics,
                name: "Graph Analytics",
                description: "Centrality measures, community detection, motifs, similarity",
                kernel_count: 15,
                feature: "graph",
            },
            DomainInfo {
                domain: Domain::StatisticalML,
                name: "Statistical ML",
                description: "Clustering, anomaly detection, regression, ensemble methods",
                kernel_count: 6,
                feature: "ml",
            },
            DomainInfo {
                domain: Domain::Compliance,
                name: "Compliance",
                description: "AML, KYC, sanctions screening, transaction monitoring",
                kernel_count: 9,
                feature: "compliance",
            },
            DomainInfo {
                domain: Domain::TemporalAnalysis,
                name: "Temporal Analysis",
                description: "Forecasting, change detection, seasonal decomposition",
                kernel_count: 7,
                feature: "temporal",
            },
            DomainInfo {
                domain: Domain::RiskAnalytics,
                name: "Risk Analytics",
                description: "Credit risk, Monte Carlo VaR, portfolio risk aggregation",
                kernel_count: 4,
                feature: "risk",
            },
            DomainInfo {
                domain: Domain::Banking,
                name: "Banking",
                description: "Fraud pattern matching with graph analysis",
                kernel_count: 1,
                feature: "banking",
            },
            DomainInfo {
                domain: Domain::BehavioralAnalytics,
                name: "Behavioral Analytics",
                description: "User profiling, anomaly profiling, forensics",
                kernel_count: 6,
                feature: "behavioral",
            },
            DomainInfo {
                domain: Domain::OrderMatching,
                name: "Order Matching",
                description: "High-frequency order book matching engine",
                kernel_count: 1,
                feature: "orderbook",
            },
            DomainInfo {
                domain: Domain::ProcessIntelligence,
                name: "Process Intelligence",
                description: "Process mining, conformance checking, DFG construction",
                kernel_count: 4,
                feature: "procint",
            },
            DomainInfo {
                domain: Domain::Clearing,
                name: "Clearing",
                description: "Settlement, DVP matching, netting calculation",
                kernel_count: 5,
                feature: "clearing",
            },
            DomainInfo {
                domain: Domain::TreasuryManagement,
                name: "Treasury Management",
                description: "Cash flow forecasting, collateral optimization, FX hedging",
                kernel_count: 5,
                feature: "treasury",
            },
            DomainInfo {
                domain: Domain::Accounting,
                name: "Accounting",
                description: "Chart of accounts mapping, reconciliation, network analysis",
                kernel_count: 5,
                feature: "accounting",
            },
            DomainInfo {
                domain: Domain::PaymentProcessing,
                name: "Payment Processing",
                description: "Transaction execution, flow analysis",
                kernel_count: 2,
                feature: "payments",
            },
            DomainInfo {
                domain: Domain::FinancialAudit,
                name: "Financial Audit",
                description: "Feature extraction, hypergraph construction",
                kernel_count: 2,
                feature: "audit",
            },
        ]
    }

    /// Get total kernel count across all domains.
    pub fn total_kernel_count() -> usize {
        domains().iter().map(|d| d.kernel_count).sum()
    }

    /// Get enabled domains based on compile-time features.
    pub fn enabled_domains() -> Vec<&'static str> {
        let mut enabled = Vec::new();

        #[cfg(feature = "graph")]
        enabled.push("graph");
        #[cfg(feature = "ml")]
        enabled.push("ml");
        #[cfg(feature = "compliance")]
        enabled.push("compliance");
        #[cfg(feature = "temporal")]
        enabled.push("temporal");
        #[cfg(feature = "risk")]
        enabled.push("risk");
        #[cfg(feature = "banking")]
        enabled.push("banking");
        #[cfg(feature = "behavioral")]
        enabled.push("behavioral");
        #[cfg(feature = "orderbook")]
        enabled.push("orderbook");
        #[cfg(feature = "procint")]
        enabled.push("procint");
        #[cfg(feature = "clearing")]
        enabled.push("clearing");
        #[cfg(feature = "treasury")]
        enabled.push("treasury");
        #[cfg(feature = "accounting")]
        enabled.push("accounting");
        #[cfg(feature = "payments")]
        enabled.push("payments");
        #[cfg(feature = "audit")]
        enabled.push("audit");

        enabled
    }
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

    #[test]
    fn test_catalog() {
        let domains = catalog::domains();
        assert_eq!(domains.len(), 14);
        assert_eq!(catalog::total_kernel_count(), 72);
    }

    #[test]
    fn test_enabled_domains() {
        let enabled = catalog::enabled_domains();
        // Default features include graph, ml, compliance, temporal, risk
        assert!(enabled.contains(&"graph"));
        assert!(enabled.contains(&"ml"));
    }
}
