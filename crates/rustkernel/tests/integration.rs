//! Integration tests for RustKernels
//!
//! These tests verify cross-domain functionality and real-world workflows.

// ============================================================================
// Catalog and Registry Tests
// ============================================================================

#[test]
fn test_catalog_domains() {
    let domains = rustkernel::catalog::domains();
    assert_eq!(domains.len(), 14, "Should have 14 domains");

    // Check that all expected domains are present
    let domain_names: Vec<&str> = domains.iter().map(|d| d.feature).collect();
    assert!(domain_names.contains(&"graph"));
    assert!(domain_names.contains(&"ml"));
    assert!(domain_names.contains(&"compliance"));
    assert!(domain_names.contains(&"temporal"));
    assert!(domain_names.contains(&"risk"));
    assert!(domain_names.contains(&"banking"));
    assert!(domain_names.contains(&"behavioral"));
    assert!(domain_names.contains(&"orderbook"));
    assert!(domain_names.contains(&"procint"));
    assert!(domain_names.contains(&"clearing"));
    assert!(domain_names.contains(&"treasury"));
    assert!(domain_names.contains(&"accounting"));
    assert!(domain_names.contains(&"payments"));
    assert!(domain_names.contains(&"audit"));
}

#[test]
fn test_catalog_kernel_count() {
    let total = rustkernel::catalog::total_kernel_count();
    assert!(total > 70, "Should have at least 70 kernels, got {}", total);
    assert!(total < 200, "Sanity check: shouldn't exceed 200 kernels");
}

#[test]
fn test_enabled_domains() {
    let enabled = rustkernel::catalog::enabled_domains();

    // Default features should include P1 domains
    assert!(enabled.contains(&"graph"), "graph should be enabled by default");
    assert!(enabled.contains(&"ml"), "ml should be enabled by default");
    assert!(enabled.contains(&"compliance"), "compliance should be enabled by default");
    assert!(enabled.contains(&"temporal"), "temporal should be enabled by default");
    assert!(enabled.contains(&"risk"), "risk should be enabled by default");
}

#[test]
fn test_domain_info_complete() {
    let domains = rustkernel::catalog::domains();

    for domain in &domains {
        assert!(!domain.name.is_empty(), "Domain name should not be empty");
        assert!(!domain.description.is_empty(), "Domain description should not be empty");
        assert!(!domain.feature.is_empty(), "Domain feature flag should not be empty");
        assert!(domain.kernel_count > 0, "Domain {} should have at least 1 kernel", domain.name);
    }
}

// ============================================================================
// Licensing System Tests
// ============================================================================

#[test]
fn test_development_license() {
    use rustkernel::core::license::{DevelopmentLicense, LicenseTier, LicenseValidator};
    use rustkernel::core::domain::Domain;

    let license = DevelopmentLicense;

    // Development license allows all domains
    for domain in Domain::ALL {
        assert!(license.validate_domain(*domain).is_ok(),
            "Development license should allow domain {:?}", domain);
    }

    // Development license allows GPU-native
    assert!(license.gpu_native_enabled());
    assert_eq!(license.tier(), LicenseTier::Development);
    assert!(license.max_kernels().is_none(), "Dev license should have unlimited kernels");
}

#[test]
fn test_community_license_restrictions() {
    use rustkernel::core::license::{License, LicenseValidator, StandardLicenseValidator};
    use rustkernel::core::domain::Domain;

    let license = License::community("Test User");
    let validator = StandardLicenseValidator::new(license);

    // Community includes Core, GraphAnalytics, StatisticalML
    assert!(validator.validate_domain(Domain::Core).is_ok());
    assert!(validator.validate_domain(Domain::GraphAnalytics).is_ok());
    assert!(validator.validate_domain(Domain::StatisticalML).is_ok());

    // Community does NOT include higher-tier domains
    assert!(validator.validate_domain(Domain::RiskAnalytics).is_err());
    assert!(validator.validate_domain(Domain::Compliance).is_err());
    assert!(validator.validate_domain(Domain::Banking).is_err());

    // Community does not support GPU-native
    assert!(!validator.gpu_native_enabled());
}

#[test]
fn test_enterprise_license() {
    use rustkernel::core::license::{License, LicenseValidator, StandardLicenseValidator};
    use rustkernel::core::domain::Domain;

    let license = License::enterprise("Enterprise Corp", None);
    let validator = StandardLicenseValidator::new(license);

    // Enterprise allows all domains
    for domain in Domain::ALL {
        assert!(validator.validate_domain(*domain).is_ok(),
            "Enterprise license should allow domain {:?}", domain);
    }

    // Enterprise supports GPU-native
    assert!(validator.gpu_native_enabled());
}

#[test]
fn test_license_guard() {
    use rustkernel::core::license::{DevelopmentLicense, LicenseGuard, LicenseValidator};
    use rustkernel::core::domain::Domain;

    let validator = DevelopmentLicense;
    let guard = LicenseGuard::new(&validator, Domain::GraphAnalytics);

    assert!(guard.check().is_ok());
    assert!(guard.check_feature("GraphAnalytics.PageRank").is_ok());
    assert!(guard.check_gpu_native().is_ok());
}

// ============================================================================
// Registry Tests
// ============================================================================

#[test]
fn test_registry_creation() {
    use rustkernel::core::registry::KernelRegistry;

    let registry = KernelRegistry::new();
    let stats = registry.stats();

    // Verify basic stats
    assert_eq!(stats.total, stats.batch_kernels + stats.ring_kernels);
}

// ============================================================================
// Graph Analytics Kernel Tests
// ============================================================================

#[cfg(feature = "graph")]
mod graph_tests {
    use rustkernel::graph::centrality::PageRank;
    use rustkernel::core::traits::GpuKernel;
    use rustkernel::core::domain::Domain;

    #[test]
    fn test_pagerank_metadata() {
        let kernel = PageRank::new();
        let metadata = kernel.metadata();

        assert!(metadata.id.contains("pagerank"), "ID should contain 'pagerank'");
        assert_eq!(metadata.domain, Domain::GraphAnalytics);
    }
}

// ============================================================================
// ML Kernel Tests
// ============================================================================

#[cfg(feature = "ml")]
mod ml_tests {
    use rustkernel::ml::clustering::KMeans;
    use rustkernel::core::traits::GpuKernel;
    use rustkernel::core::domain::Domain;

    #[test]
    fn test_kmeans_metadata() {
        let kernel = KMeans::new();
        let metadata = kernel.metadata();

        assert!(metadata.id.contains("kmeans"), "ID should contain 'kmeans'");
        assert_eq!(metadata.domain, Domain::StatisticalML);
    }
}

// ============================================================================
// Compliance Kernel Tests
// ============================================================================

#[cfg(feature = "compliance")]
mod compliance_tests {
    use rustkernel::compliance::aml::CircularFlowRatio;
    use rustkernel::core::traits::GpuKernel;
    use rustkernel::core::domain::Domain;

    #[test]
    fn test_circular_flow_metadata() {
        let kernel = CircularFlowRatio::new();
        let metadata = kernel.metadata();

        assert!(metadata.id.contains("circular"), "ID should contain 'circular'");
        assert_eq!(metadata.domain, Domain::Compliance);
    }
}

// ============================================================================
// Temporal Kernel Tests
// ============================================================================

#[cfg(feature = "temporal")]
mod temporal_tests {
    use rustkernel::temporal::forecasting::ARIMAForecast;
    use rustkernel::core::traits::GpuKernel;
    use rustkernel::core::domain::Domain;

    #[test]
    fn test_arima_metadata() {
        let kernel = ARIMAForecast::new();
        let metadata = kernel.metadata();

        assert!(metadata.id.contains("arima"), "ID should contain 'arima'");
        assert_eq!(metadata.domain, Domain::TemporalAnalysis);
    }
}

// ============================================================================
// Risk Kernel Tests
// ============================================================================

#[cfg(feature = "risk")]
mod risk_tests {
    use rustkernel::risk::market::MonteCarloVaR;
    use rustkernel::core::traits::GpuKernel;
    use rustkernel::core::domain::Domain;

    #[test]
    fn test_monte_carlo_var_metadata() {
        let kernel = MonteCarloVaR::new();
        let metadata = kernel.metadata();

        assert!(metadata.id.contains("monte") || metadata.id.contains("var"),
            "ID should contain 'monte' or 'var'");
        assert_eq!(metadata.domain, Domain::RiskAnalytics);
    }
}

// ============================================================================
// Version and Feature Tests
// ============================================================================

#[test]
fn test_version_info() {
    assert!(!rustkernel::version::VERSION.is_empty());
    assert!(!rustkernel::version::MIN_RINGKERNEL_VERSION.is_empty());
}

#[test]
fn test_prelude_exports() {
    // Verify prelude exports core types
    use rustkernel::prelude::*;

    let _ = Domain::GraphAnalytics;
    let _ = KernelMode::Ring;
    let _ = KernelMode::Batch;
}

// ============================================================================
// BatchKernel Integration Tests
// ============================================================================

#[cfg(feature = "graph")]
mod batch_kernel_graph_tests {
    use rustkernel::core::traits::BatchKernel;
    use rustkernel::graph::centrality::PageRank;
    use rustkernel::graph::messages::{CentralityInput, CentralityOutput, CentralityParams};
    use rustkernel::graph::types::CsrGraph;

    #[tokio::test]
    async fn test_pagerank_batch_execution() {
        let kernel = PageRank::new();

        // Create a simple graph: 0 -> 1 -> 2 -> 0 (cycle)
        // CSR format: row_offsets, col_indices, values
        let graph = CsrGraph::from_edges(3, &[(0, 1), (1, 2), (2, 0)]);

        let input = CentralityInput {
            graph,
            normalize: true,
            max_iterations: Some(100),
            tolerance: Some(1e-6),
            params: CentralityParams::PageRank { damping: 0.85 },
        };

        let result: CentralityOutput = kernel.execute(input).await.expect("PageRank should execute");

        assert!(!result.result.scores.is_empty(), "Should have scores");
        assert!(result.compute_time_us > 0, "Should record compute time");

        // In a cycle, all nodes should have similar scores
        let scores: Vec<f64> = result.result.scores.iter().map(|ns| ns.score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        for score in &scores {
            assert!((score - mean).abs() < 0.1, "Scores in cycle should be similar");
        }
    }
}

#[cfg(feature = "ml")]
mod batch_kernel_ml_tests {
    use rustkernel::core::traits::BatchKernel;
    use rustkernel::ml::clustering::KMeans;
    use rustkernel::ml::messages::{KMeansInput, KMeansOutput};
    use rustkernel::ml::types::DataMatrix;

    #[tokio::test]
    async fn test_kmeans_batch_execution() {
        let kernel = KMeans::new();

        // Create simple 2D data with 2 obvious clusters
        let data = DataMatrix::from_rows(&[
            &[0.0, 0.0],
            &[0.1, 0.1],
            &[0.0, 0.1],
            &[10.0, 10.0],
            &[10.1, 10.1],
            &[10.0, 10.1],
        ]);

        let input = KMeansInput::new(data, 2)
            .with_max_iterations(100)
            .with_tolerance(1e-6);

        let result: KMeansOutput = kernel.execute(input).await.expect("KMeans should execute");

        assert_eq!(result.result.labels.len(), 6, "Should have 6 assignments");
        assert_eq!(result.result.n_clusters, 2, "Should have 2 clusters");
        assert!(result.compute_time_us > 0, "Should record compute time");

        // First 3 points should be in same cluster, last 3 in another
        assert_eq!(result.result.labels[0], result.result.labels[1]);
        assert_eq!(result.result.labels[1], result.result.labels[2]);
        assert_eq!(result.result.labels[3], result.result.labels[4]);
        assert_eq!(result.result.labels[4], result.result.labels[5]);
        assert_ne!(result.result.labels[0], result.result.labels[3], "Clusters should be different");
    }
}

#[cfg(feature = "risk")]
mod batch_kernel_risk_tests {
    use rustkernel::core::traits::BatchKernel;
    use rustkernel::risk::credit::CreditRiskScoring;
    use rustkernel::risk::messages::{CreditRiskScoringInput, CreditRiskScoringOutput};
    use rustkernel::risk::types::CreditFactors;

    #[tokio::test]
    async fn test_credit_risk_batch_execution() {
        let kernel = CreditRiskScoring::new();

        let factors = CreditFactors {
            obligor_id: 1,
            debt_to_income: 0.30,
            loan_to_value: 0.70,
            credit_utilization: 0.25,
            payment_history: 85.0,
            employment_years: 5.0,
            recent_inquiries: 2,
            delinquencies: 0,
            credit_history_years: 8.0,
        };

        let input = CreditRiskScoringInput::new(factors, 100_000.0, 5.0);

        let result: CreditRiskScoringOutput = kernel.execute(input).await.expect("CreditRisk should execute");

        // Verify PD is in valid range [0, 1]
        assert!(result.result.pd >= 0.0);
        assert!(result.result.pd <= 1.0);

        // Verify LGD is in valid range [0, 1]
        assert!(result.result.lgd >= 0.0);
        assert!(result.result.lgd <= 1.0);

        // Verify expected loss is positive
        assert!(result.result.expected_loss >= 0.0);

        assert!(result.compute_time_us > 0, "Should record compute time");
    }
}

#[cfg(feature = "temporal")]
mod batch_kernel_temporal_tests {
    use rustkernel::core::traits::BatchKernel;
    use rustkernel::temporal::forecasting::ARIMAForecast;
    use rustkernel::temporal::messages::{ARIMAForecastInput, ARIMAForecastOutput};
    use rustkernel::temporal::types::{ARIMAParams, TimeSeries};

    #[tokio::test]
    async fn test_arima_batch_execution() {
        let kernel = ARIMAForecast::new();

        // Create simple time series data
        let values: Vec<f64> = (0..100).map(|i| 10.0 + 0.1 * i as f64 + (i as f64 * 0.1).sin()).collect();
        let series = TimeSeries::new(values);
        let params = ARIMAParams::new(1, 1, 1);

        let input = ARIMAForecastInput::new(series, params, 10);

        let result: ARIMAForecastOutput = kernel.execute(input).await.expect("ARIMA should execute");

        assert_eq!(result.result.forecast.len(), 10, "Should have 10 forecasts");
        assert!(result.compute_time_us > 0, "Should record compute time");
    }
}

#[cfg(feature = "orderbook")]
mod batch_kernel_orderbook_tests {
    use rustkernel::core::traits::BatchKernel;
    use rustkernel::orderbook::matching::OrderMatchingEngine;
    use rustkernel::orderbook::messages::{SubmitOrderInput, SubmitOrderOutput, BatchOrderInput, BatchOrderOutput};
    use rustkernel::orderbook::types::{Order, Side, Price, Quantity, OrderStatus};

    #[tokio::test]
    async fn test_order_matching_single_order() {
        let engine = OrderMatchingEngine::new();

        let order = Order::limit(1, 1, Side::Buy, Price::from_f64(100.0), Quantity::from_f64(10.0), 100, 1);
        let input = SubmitOrderInput::new(order);

        let result: SubmitOrderOutput = engine.execute(input).await.expect("Submit should execute");

        // Order should be accepted (no matching sells)
        assert_eq!(result.result.status, OrderStatus::New);
        assert_eq!(result.result.remaining, Quantity::from_f64(10.0));
        assert!(result.compute_time_us > 0, "Should record compute time");
    }

    #[tokio::test]
    async fn test_order_matching_batch() {
        let engine = OrderMatchingEngine::new();

        // Submit a buy and matching sell
        let buy = Order::limit(1, 1, Side::Buy, Price::from_f64(100.0), Quantity::from_f64(10.0), 100, 1);
        let sell = Order::limit(2, 1, Side::Sell, Price::from_f64(100.0), Quantity::from_f64(10.0), 200, 2);

        let input = BatchOrderInput::new(vec![buy, sell]);

        let result: BatchOrderOutput = engine.execute(input).await.expect("Batch should execute");

        assert_eq!(result.results.len(), 2, "Should have 2 results");
        assert!(result.total_trades >= 1, "Should have at least 1 trade");
        assert!(result.compute_time_us > 0, "Should record compute time");
    }
}

#[cfg(feature = "compliance")]
mod batch_kernel_compliance_tests {
    use rustkernel::core::traits::BatchKernel;
    use rustkernel::compliance::aml::CircularFlowRatio;
    use rustkernel::compliance::messages::{CircularFlowInput, CircularFlowOutput};
    use rustkernel::compliance::types::Transaction;

    #[tokio::test]
    async fn test_circular_flow_detection() {
        let kernel = CircularFlowRatio::new();

        // Create transactions that form a cycle: A -> B -> C -> A
        let transactions = vec![
            Transaction {
                id: 1,
                source_id: 1,
                dest_id: 2,
                amount: 1000.0,
                timestamp: 1000,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 2,
                source_id: 2,
                dest_id: 3,
                amount: 900.0,
                timestamp: 2000,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
            Transaction {
                id: 3,
                source_id: 3,
                dest_id: 1,
                amount: 800.0,
                timestamp: 3000,
                currency: "USD".to_string(),
                tx_type: "wire".to_string(),
            },
        ];

        let input = CircularFlowInput::new(transactions, 100.0);

        let result: CircularFlowOutput = kernel.execute(input).await.expect("CircularFlow should execute");

        // Should detect the cycle
        assert!(result.result.circular_ratio > 0.0, "Should detect circular flow");
        assert!(!result.result.sccs.is_empty(), "Should find strongly connected components");
        assert!(result.compute_time_us > 0, "Should record compute time");
    }
}
