//! Domain definitions for kernel categorization.
//!
//! Kernels are organized into domains representing different business/analytical areas.
//! Domains are used for:
//! - License enforcement (licensing per domain)
//! - Kernel discovery and organization
//! - Feature gating

use serde::{Deserialize, Serialize};
use std::fmt;

/// Business/analytical domain for kernel categorization.
///
/// Each domain represents a distinct area of functionality:
/// - Financial services (banking, compliance, risk, treasury)
/// - Analytics (graph, ML, temporal, behavioral)
/// - Operations (clearing, payments, order matching)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Domain {
    /// Graph analytics: centrality, community detection, motifs, similarity
    GraphAnalytics,

    /// Statistical machine learning: clustering, anomaly detection, regression
    StatisticalML,

    /// Compliance: AML, KYC, sanctions screening, transaction monitoring
    Compliance,

    /// Temporal analysis: forecasting, change detection, decomposition
    TemporalAnalysis,

    /// Risk analytics: credit risk, VaR, portfolio risk, stress testing
    RiskAnalytics,

    /// Banking: fraud detection and pattern matching
    Banking,

    /// Behavioral analytics: profiling, forensics, event correlation
    BehavioralAnalytics,

    /// Order matching: high-frequency order book matching
    OrderMatching,

    /// Process intelligence: process mining, conformance checking
    ProcessIntelligence,

    /// Clearing: settlement, netting, DVP matching
    Clearing,

    /// Treasury management: cash flow, collateral, hedging, liquidity
    TreasuryManagement,

    /// Accounting: chart of accounts, journal transformation, reconciliation
    Accounting,

    /// Payment processing: transaction execution, flow analysis
    PaymentProcessing,

    /// Financial audit: feature extraction, hypergraph construction
    FinancialAudit,

    /// Core: test kernels and infrastructure validation
    #[default]
    Core,
}

impl Domain {
    /// All available domains.
    pub const ALL: &'static [Domain] = &[
        Domain::GraphAnalytics,
        Domain::StatisticalML,
        Domain::Compliance,
        Domain::TemporalAnalysis,
        Domain::RiskAnalytics,
        Domain::Banking,
        Domain::BehavioralAnalytics,
        Domain::OrderMatching,
        Domain::ProcessIntelligence,
        Domain::Clearing,
        Domain::TreasuryManagement,
        Domain::Accounting,
        Domain::PaymentProcessing,
        Domain::FinancialAudit,
        Domain::Core,
    ];

    /// Priority 1 domains (high-value, implement first).
    pub const P1: &'static [Domain] = &[
        Domain::GraphAnalytics,
        Domain::StatisticalML,
        Domain::Compliance,
        Domain::TemporalAnalysis,
        Domain::RiskAnalytics,
    ];

    /// Priority 2 domains (medium priority).
    pub const P2: &'static [Domain] = &[
        Domain::Banking,
        Domain::BehavioralAnalytics,
        Domain::OrderMatching,
        Domain::ProcessIntelligence,
        Domain::Clearing,
    ];

    /// Priority 3 domains (lower priority).
    pub const P3: &'static [Domain] = &[
        Domain::TreasuryManagement,
        Domain::Accounting,
        Domain::PaymentProcessing,
        Domain::FinancialAudit,
    ];

    /// Returns the domain name as a string slice.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Domain::GraphAnalytics => "GraphAnalytics",
            Domain::StatisticalML => "StatisticalML",
            Domain::Compliance => "Compliance",
            Domain::TemporalAnalysis => "TemporalAnalysis",
            Domain::RiskAnalytics => "RiskAnalytics",
            Domain::Banking => "Banking",
            Domain::BehavioralAnalytics => "BehavioralAnalytics",
            Domain::OrderMatching => "OrderMatching",
            Domain::ProcessIntelligence => "ProcessIntelligence",
            Domain::Clearing => "Clearing",
            Domain::TreasuryManagement => "TreasuryManagement",
            Domain::Accounting => "Accounting",
            Domain::PaymentProcessing => "PaymentProcessing",
            Domain::FinancialAudit => "FinancialAudit",
            Domain::Core => "Core",
        }
    }

    /// Parse a domain from a string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "GraphAnalytics" => Some(Domain::GraphAnalytics),
            "StatisticalML" => Some(Domain::StatisticalML),
            "Compliance" => Some(Domain::Compliance),
            "TemporalAnalysis" => Some(Domain::TemporalAnalysis),
            "RiskAnalytics" => Some(Domain::RiskAnalytics),
            "Banking" => Some(Domain::Banking),
            "BehavioralAnalytics" => Some(Domain::BehavioralAnalytics),
            "OrderMatching" => Some(Domain::OrderMatching),
            "ProcessIntelligence" => Some(Domain::ProcessIntelligence),
            "Clearing" => Some(Domain::Clearing),
            "TreasuryManagement" => Some(Domain::TreasuryManagement),
            "Accounting" => Some(Domain::Accounting),
            "PaymentProcessing" => Some(Domain::PaymentProcessing),
            "FinancialAudit" => Some(Domain::FinancialAudit),
            "Core" => Some(Domain::Core),
            _ => None,
        }
    }

    /// Returns true if this is a P1 (high priority) domain.
    #[must_use]
    pub const fn is_p1(&self) -> bool {
        matches!(
            self,
            Domain::GraphAnalytics
                | Domain::StatisticalML
                | Domain::Compliance
                | Domain::TemporalAnalysis
                | Domain::RiskAnalytics
        )
    }

    /// Convert to the corresponding `ringkernel_core::domain::Domain` variant.
    ///
    /// Mapping notes:
    /// - `TemporalAnalysis` maps to `TimeSeries`
    /// - `RiskAnalytics` maps to `RiskManagement`
    /// - `Core` maps to `General`
    /// - All other variants map by name.
    #[must_use]
    pub fn to_ring_domain(&self) -> ringkernel_core::domain::Domain {
        ringkernel_core::domain::Domain::from(*self)
    }

    /// Construct a `Domain` from a `ringkernel_core::domain::Domain` variant.
    ///
    /// Mapping notes:
    /// - `TimeSeries` maps to `TemporalAnalysis`
    /// - `RiskManagement` maps to `RiskAnalytics`
    /// - `General` maps to `Core`
    /// - Variants without a direct counterpart (`MarketData`, `Settlement`,
    ///   `NetworkAnalysis`, `FraudDetection`, `Simulation`, `Custom`) are
    ///   mapped to the closest match or `Core`.
    #[must_use]
    pub fn from_ring_domain(ring: ringkernel_core::domain::Domain) -> Self {
        Domain::from(ring)
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Bidirectional conversion: Domain <-> ringkernel_core::domain::Domain (0.4.2)
// ---------------------------------------------------------------------------

impl From<Domain> for ringkernel_core::domain::Domain {
    fn from(d: Domain) -> Self {
        use ringkernel_core::domain::Domain as RD;
        match d {
            Domain::GraphAnalytics => RD::GraphAnalytics,
            Domain::StatisticalML => RD::StatisticalML,
            Domain::Compliance => RD::Compliance,
            Domain::TemporalAnalysis => RD::TimeSeries,
            Domain::RiskAnalytics => RD::RiskManagement,
            Domain::Banking => RD::Banking,
            Domain::BehavioralAnalytics => RD::BehavioralAnalytics,
            Domain::OrderMatching => RD::OrderMatching,
            Domain::ProcessIntelligence => RD::ProcessIntelligence,
            Domain::Clearing => RD::Clearing,
            Domain::TreasuryManagement => RD::TreasuryManagement,
            Domain::Accounting => RD::Accounting,
            Domain::PaymentProcessing => RD::PaymentProcessing,
            Domain::FinancialAudit => RD::FinancialAudit,
            Domain::Core => RD::General,
        }
    }
}

impl From<ringkernel_core::domain::Domain> for Domain {
    fn from(rd: ringkernel_core::domain::Domain) -> Self {
        use ringkernel_core::domain::Domain as RD;
        match rd {
            RD::General => Domain::Core,
            RD::GraphAnalytics => Domain::GraphAnalytics,
            RD::StatisticalML => Domain::StatisticalML,
            RD::Compliance => Domain::Compliance,
            RD::RiskManagement => Domain::RiskAnalytics,
            RD::OrderMatching => Domain::OrderMatching,
            RD::Accounting => Domain::Accounting,
            RD::TimeSeries => Domain::TemporalAnalysis,
            RD::Banking => Domain::Banking,
            RD::BehavioralAnalytics => Domain::BehavioralAnalytics,
            RD::ProcessIntelligence => Domain::ProcessIntelligence,
            RD::Clearing => Domain::Clearing,
            RD::TreasuryManagement => Domain::TreasuryManagement,
            RD::PaymentProcessing => Domain::PaymentProcessing,
            RD::FinancialAudit => Domain::FinancialAudit,
            // Variants present in ringkernel_core but without a direct
            // counterpart in this crate are mapped to the closest match.
            RD::NetworkAnalysis => Domain::GraphAnalytics,
            RD::FraudDetection => Domain::Banking,
            RD::Settlement => Domain::Clearing,
            RD::MarketData => Domain::Core,
            RD::Simulation => Domain::Core,
            RD::Custom => Domain::Core,
            // Future-proofing: map unknown variants to Core.
            _ => Domain::Core,
        }
    }
}

/// Feature strings for fine-grained licensing.
///
/// Format: `Domain.Feature` (e.g., `GraphAnalytics.PageRank`)
pub mod features {
    // GraphAnalytics domain
    /// PageRank centrality calculation
    pub const GRAPH_PAGERANK: &str = "GraphAnalytics.PageRank";
    /// Betweenness centrality calculation
    pub const GRAPH_BETWEENNESS: &str = "GraphAnalytics.BetweennessCentrality";
    /// Closeness centrality calculation
    pub const GRAPH_CLOSENESS: &str = "GraphAnalytics.ClosenessCentrality";
    /// Degree centrality calculation
    pub const GRAPH_DEGREE: &str = "GraphAnalytics.DegreeCentrality";
    /// Eigenvector centrality calculation
    pub const GRAPH_EIGENVECTOR: &str = "GraphAnalytics.EigenvectorCentrality";
    /// Katz centrality calculation
    pub const GRAPH_KATZ: &str = "GraphAnalytics.KatzCentrality";
    /// Community detection algorithms
    pub const GRAPH_COMMUNITY: &str = "GraphAnalytics.CommunityDetection";
    /// Motif detection algorithms
    pub const GRAPH_MOTIF: &str = "GraphAnalytics.MotifDetection";
    /// Similarity metrics
    pub const GRAPH_SIMILARITY: &str = "GraphAnalytics.Similarity";

    // StatisticalML domain
    /// K-Means clustering
    pub const ML_KMEANS: &str = "StatisticalML.KMeans";
    /// DBSCAN clustering
    pub const ML_DBSCAN: &str = "StatisticalML.DBSCAN";
    /// Isolation Forest anomaly detection
    pub const ML_ISOLATION_FOREST: &str = "StatisticalML.IsolationForest";
    /// Local Outlier Factor anomaly detection
    pub const ML_LOF: &str = "StatisticalML.LocalOutlierFactor";
    /// Ensemble methods
    pub const ML_ENSEMBLE: &str = "StatisticalML.Ensemble";
    /// Regression models
    pub const ML_REGRESSION: &str = "StatisticalML.Regression";

    // Compliance domain
    /// Anti-Money Laundering pattern detection
    pub const COMPLIANCE_AML: &str = "Compliance.AML";
    /// Sanctions screening
    pub const COMPLIANCE_SANCTIONS: &str = "Compliance.SanctionsScreening";
    /// Know Your Customer scoring
    pub const COMPLIANCE_KYC: &str = "Compliance.KYC";
    /// Transaction monitoring
    pub const COMPLIANCE_MONITORING: &str = "Compliance.TransactionMonitoring";

    // RiskAnalytics domain
    /// Credit risk scoring
    pub const RISK_CREDIT: &str = "RiskAnalytics.CreditRisk";
    /// Market risk / VaR
    pub const RISK_MARKET: &str = "RiskAnalytics.MarketRisk";
    /// Value at Risk calculation
    pub const RISK_VAR: &str = "RiskAnalytics.VaR";
    /// Portfolio risk aggregation
    pub const RISK_PORTFOLIO: &str = "RiskAnalytics.PortfolioRisk";
    /// Stress testing
    pub const RISK_STRESS: &str = "RiskAnalytics.StressTesting";

    // TemporalAnalysis domain
    /// ARIMA forecasting
    pub const TEMPORAL_ARIMA: &str = "TemporalAnalysis.ARIMA";
    /// Prophet-style decomposition
    pub const TEMPORAL_PROPHET: &str = "TemporalAnalysis.Prophet";
    /// Change point detection
    pub const TEMPORAL_CHANGEPOINT: &str = "TemporalAnalysis.ChangePoint";
    /// Seasonal decomposition
    pub const TEMPORAL_SEASONAL: &str = "TemporalAnalysis.Seasonal";
    /// Volatility analysis
    pub const TEMPORAL_VOLATILITY: &str = "TemporalAnalysis.Volatility";

    // Banking domain
    /// Fraud detection
    pub const BANKING_FRAUD: &str = "Banking.FraudDetection";

    // OrderMatching domain
    /// Order book matching
    pub const ORDERBOOK_MATCHING: &str = "OrderMatching.OrderMatching";

    // ProcessIntelligence domain
    /// Process mining
    pub const PROCINT_MINING: &str = "ProcessIntelligence.ProcessMining";
    /// Conformance checking
    pub const PROCINT_CONFORMANCE: &str = "ProcessIntelligence.ConformanceChecking";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_all_count() {
        assert_eq!(Domain::ALL.len(), 15);
    }

    #[test]
    fn test_domain_priority_coverage() {
        let p1_count = Domain::P1.len();
        let p2_count = Domain::P2.len();
        let p3_count = Domain::P3.len();
        // Core is P0, not in P1/P2/P3
        assert_eq!(p1_count + p2_count + p3_count + 1, Domain::ALL.len());
    }

    #[test]
    fn test_domain_from_str() {
        assert_eq!(
            Domain::parse("GraphAnalytics"),
            Some(Domain::GraphAnalytics)
        );
        assert_eq!(Domain::parse("Unknown"), None);
    }

    #[test]
    fn test_domain_display() {
        assert_eq!(Domain::GraphAnalytics.to_string(), "GraphAnalytics");
        assert_eq!(Domain::RiskAnalytics.to_string(), "RiskAnalytics");
    }

    #[test]
    fn test_domain_default() {
        assert_eq!(Domain::default(), Domain::Core);
    }

    #[test]
    fn test_to_ring_domain_renamed_variants() {
        use ringkernel_core::domain::Domain as RD;

        assert_eq!(Domain::TemporalAnalysis.to_ring_domain(), RD::TimeSeries);
        assert_eq!(Domain::RiskAnalytics.to_ring_domain(), RD::RiskManagement);
        assert_eq!(Domain::Core.to_ring_domain(), RD::General);
    }

    #[test]
    fn test_to_ring_domain_identity_variants() {
        use ringkernel_core::domain::Domain as RD;

        assert_eq!(Domain::GraphAnalytics.to_ring_domain(), RD::GraphAnalytics);
        assert_eq!(Domain::StatisticalML.to_ring_domain(), RD::StatisticalML);
        assert_eq!(Domain::Compliance.to_ring_domain(), RD::Compliance);
        assert_eq!(Domain::Banking.to_ring_domain(), RD::Banking);
        assert_eq!(
            Domain::BehavioralAnalytics.to_ring_domain(),
            RD::BehavioralAnalytics
        );
        assert_eq!(Domain::OrderMatching.to_ring_domain(), RD::OrderMatching);
        assert_eq!(
            Domain::ProcessIntelligence.to_ring_domain(),
            RD::ProcessIntelligence
        );
        assert_eq!(Domain::Clearing.to_ring_domain(), RD::Clearing);
        assert_eq!(
            Domain::TreasuryManagement.to_ring_domain(),
            RD::TreasuryManagement
        );
        assert_eq!(Domain::Accounting.to_ring_domain(), RD::Accounting);
        assert_eq!(
            Domain::PaymentProcessing.to_ring_domain(),
            RD::PaymentProcessing
        );
        assert_eq!(Domain::FinancialAudit.to_ring_domain(), RD::FinancialAudit);
    }

    #[test]
    fn test_from_ring_domain_renamed_variants() {
        use ringkernel_core::domain::Domain as RD;

        assert_eq!(
            Domain::from_ring_domain(RD::TimeSeries),
            Domain::TemporalAnalysis
        );
        assert_eq!(
            Domain::from_ring_domain(RD::RiskManagement),
            Domain::RiskAnalytics
        );
        assert_eq!(Domain::from_ring_domain(RD::General), Domain::Core);
    }

    #[test]
    fn test_from_ring_domain_closest_match() {
        use ringkernel_core::domain::Domain as RD;

        assert_eq!(
            Domain::from_ring_domain(RD::NetworkAnalysis),
            Domain::GraphAnalytics
        );
        assert_eq!(
            Domain::from_ring_domain(RD::FraudDetection),
            Domain::Banking
        );
        assert_eq!(Domain::from_ring_domain(RD::Settlement), Domain::Clearing);
        assert_eq!(Domain::from_ring_domain(RD::MarketData), Domain::Core);
        assert_eq!(Domain::from_ring_domain(RD::Simulation), Domain::Core);
        assert_eq!(Domain::from_ring_domain(RD::Custom), Domain::Core);
    }

    #[test]
    fn test_ring_domain_roundtrip() {
        // Every rustkernel Domain should survive a roundtrip through
        // ringkernel_core and back.
        for &domain in Domain::ALL {
            let ring = domain.to_ring_domain();
            let back = Domain::from_ring_domain(ring);
            assert_eq!(
                back, domain,
                "roundtrip failed for {:?} -> {:?} -> {:?}",
                domain, ring, back
            );
        }
    }
}
