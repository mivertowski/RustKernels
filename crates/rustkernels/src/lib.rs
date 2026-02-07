//! # RustKernels
//!
//! GPU-accelerated kernel library for financial services, analytics, and compliance workloads.
//!
//! RustKernels is built on [RingKernel 0.4.2](https://crates.io/crates/ringkernel-core)
//! and provides 106 kernels across 14 domain-specific crates.
//!
//! ## Features
//!
//! - **14 Domain Categories**: Graph analytics, ML, compliance, risk, temporal analysis, and more
//! - **106 Kernels**: Comprehensive coverage of financial and analytical algorithms
//! - **Dual Execution Modes**:
//!   - **Batch**: CPU-orchestrated, 10-50μs overhead, for periodic heavy computation
//!   - **Ring**: GPU-persistent actor, 100-500ns latency, for high-frequency operations
//! - **Type-Erased Execution**: REST/gRPC dispatch via `TypeErasedBatchKernel`
//! - **Enterprise Features**: Security, observability, resilience, production configuration
//! - **Multi-Backend**: CUDA, WebGPU, and CPU backends via RingKernel
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rustkernels::prelude::*;
//! use rustkernels::graph::centrality::{BetweennessCentrality, BetweennessCentralityInput};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let kernel = BetweennessCentrality::new();
//!
//!     let input = BetweennessCentralityInput {
//!         num_nodes: 4,
//!         edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
//!         normalized: true,
//!     };
//!
//!     let result = kernel.execute(input).await?;
//!     for (node, score) in result.scores.iter().enumerate() {
//!         println!("Node {}: {:.4}", node, score);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Domain Organization
//!
//! Kernels are organized into 14 domains:
//!
//! | Domain | Kernels | Description |
//! |--------|---------|-------------|
//! | GraphAnalytics | 28 | Centrality, GNN inference, community detection |
//! | StatisticalML | 17 | Clustering, NLP embeddings, federated learning |
//! | Compliance | 11 | AML, KYC, sanctions screening |
//! | TemporalAnalysis | 7 | Forecasting, change-point detection |
//! | RiskAnalytics | 5 | Credit risk, VaR, stress testing |
//! | ProcessIntelligence | 7 | DFG, conformance, digital twin |
//! | BehavioralAnalytics | 6 | Profiling, forensics |
//! | Clearing | 5 | Netting, settlement, DVP |
//! | Treasury | 5 | Cash flow, FX hedging |
//! | Accounting | 9 | Network generation, reconciliation |
//! | Payments | 2 | Payment processing |
//! | Banking | 1 | Fraud detection |
//! | OrderMatching | 1 | HFT order book |
//! | FinancialAudit | 2 | Feature extraction |
//!
//! ## Feature Flags
//!
//! ```toml
//! [dependencies]
//! rustkernels = { version = "0.4", features = ["graph", "ml", "risk"] }
//! ```
//!
//! Available features:
//! - `default`: graph, ml, compliance, temporal, risk
//! - `full`: All 14 domains
//! - Individual: `graph`, `ml`, `compliance`, `temporal`, `risk`, `banking`, etc.

#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export core crate
pub use rustkernel_core as core;

// Re-export derive macros
pub use rustkernel_derive::{KernelMessage, gpu_kernel, kernel_state};

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
    pub use rustkernel_derive::{KernelMessage, gpu_kernel, kernel_state};

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

    /// Minimum supported RingKernel version.
    pub const MIN_RINGKERNEL_VERSION: &str = "0.4.2";
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
                description: "Centrality, GNN inference, community detection, similarity, topology",
                kernel_count: 28,
                feature: "graph",
            },
            DomainInfo {
                domain: Domain::StatisticalML,
                name: "Statistical ML",
                description: "Clustering, NLP embeddings, federated learning, anomaly detection, explainability",
                kernel_count: 17,
                feature: "ml",
            },
            DomainInfo {
                domain: Domain::Compliance,
                name: "Compliance",
                description: "AML pattern detection, KYC scoring, sanctions screening, transaction monitoring",
                kernel_count: 11,
                feature: "compliance",
            },
            DomainInfo {
                domain: Domain::TemporalAnalysis,
                name: "Temporal Analysis",
                description: "ARIMA, Prophet decomposition, change-point detection, seasonal decomposition",
                kernel_count: 7,
                feature: "temporal",
            },
            DomainInfo {
                domain: Domain::RiskAnalytics,
                name: "Risk Analytics",
                description: "Credit risk, Monte Carlo VaR, portfolio risk, stress testing, correlation",
                kernel_count: 5,
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
                description: "Profiling, anomaly profiling, forensics, event correlation",
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
                description: "DFG construction, conformance checking, digital twin, next-activity prediction",
                kernel_count: 7,
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
                description: "Network generation, reconciliation, duplicate detection, currency conversion",
                kernel_count: 9,
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

    /// Check if a specific domain is enabled via compile-time features.
    pub fn is_domain_enabled(feature: &str) -> bool {
        enabled_domains().contains(&feature)
    }

    /// Get enabled domains based on compile-time features.
    #[allow(clippy::vec_init_then_push)]
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

/// Register all enabled domain kernels into a registry.
///
/// This is the primary entrypoint for populating a `KernelRegistry` with
/// all available kernels based on compile-time feature flags.
///
/// # Errors
///
/// Returns an error if any kernel registration fails.
pub fn register_all(
    registry: &rustkernel_core::registry::KernelRegistry,
) -> rustkernel_core::error::Result<()> {
    #[cfg(feature = "graph")]
    rustkernel_graph::register_all(registry)?;

    #[cfg(feature = "ml")]
    rustkernel_ml::register_all(registry)?;

    #[cfg(feature = "compliance")]
    rustkernel_compliance::register_all(registry)?;

    #[cfg(feature = "temporal")]
    rustkernel_temporal::register_all(registry)?;

    #[cfg(feature = "risk")]
    rustkernel_risk::register_all(registry)?;

    #[cfg(feature = "banking")]
    rustkernel_banking::register_all(registry)?;

    #[cfg(feature = "behavioral")]
    rustkernel_behavioral::register_all(registry)?;

    #[cfg(feature = "orderbook")]
    rustkernel_orderbook::register_all(registry)?;

    #[cfg(feature = "procint")]
    rustkernel_procint::register_all(registry)?;

    #[cfg(feature = "clearing")]
    rustkernel_clearing::register_all(registry)?;

    #[cfg(feature = "treasury")]
    rustkernel_treasury::register_all(registry)?;

    #[cfg(feature = "accounting")]
    rustkernel_accounting::register_all(registry)?;

    #[cfg(feature = "payments")]
    rustkernel_payments::register_all(registry)?;

    #[cfg(feature = "audit")]
    rustkernel_audit::register_all(registry)?;

    Ok(())
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
    #[allow(clippy::const_is_empty)]
    fn test_version() {
        assert!(!version::VERSION.is_empty());
        assert_eq!(version::MIN_RINGKERNEL_VERSION, "0.4.2");
    }

    #[test]
    fn test_catalog() {
        let domains = catalog::domains();
        assert_eq!(domains.len(), 14);
        assert_eq!(catalog::total_kernel_count(), 106);
    }

    #[test]
    fn test_enabled_domains() {
        let enabled = catalog::enabled_domains();
        // Default features include graph, ml, compliance, temporal, risk
        assert!(enabled.contains(&"graph"));
        assert!(enabled.contains(&"ml"));
    }

    #[test]
    fn test_register_all() {
        let registry = rustkernel_core::registry::KernelRegistry::new();
        register_all(&registry).unwrap();
        // Verify kernels were registered (at least default features)
        assert!(registry.total_count() > 0);
    }
}
