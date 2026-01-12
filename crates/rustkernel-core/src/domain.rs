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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    pub fn from_str(s: &str) -> Option<Self> {
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
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
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
        assert_eq!(Domain::from_str("GraphAnalytics"), Some(Domain::GraphAnalytics));
        assert_eq!(Domain::from_str("Unknown"), None);
    }

    #[test]
    fn test_domain_display() {
        assert_eq!(Domain::GraphAnalytics.to_string(), "GraphAnalytics");
        assert_eq!(Domain::RiskAnalytics.to_string(), "RiskAnalytics");
    }
}
