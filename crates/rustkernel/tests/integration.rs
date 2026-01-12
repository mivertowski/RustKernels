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
