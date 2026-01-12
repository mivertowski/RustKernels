//! Licensing and feature gating system.
//!
//! This module provides enterprise licensing infrastructure similar to the
//! C# Orleans.GpuBridge.Kernels licensing system.
//!
//! # License Enforcement Points
//!
//! 1. At kernel registration time (via `KernelRegistry`)
//! 2. At actor activation time (like Orleans `OnActivateAsync`)
//! 3. At runtime via `LicenseGuard`

use crate::domain::Domain;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// License validation errors.
#[derive(Debug, Error, Clone)]
pub enum LicenseError {
    /// Domain is not licensed.
    #[error("Domain '{0}' is not licensed")]
    DomainNotLicensed(Domain),

    /// Feature is not licensed.
    #[error("Feature '{0}' is not licensed")]
    FeatureNotLicensed(String),

    /// GPU-native kernels require Enterprise license.
    #[error("GPU-native kernels require Enterprise license")]
    GpuNativeNotLicensed,

    /// License has expired.
    #[error("License expired at {0}")]
    Expired(DateTime<Utc>),

    /// Maximum kernel count exceeded.
    #[error("Maximum kernel count ({0}) exceeded")]
    KernelLimitExceeded(usize),

    /// License validation failed.
    #[error("License validation failed: {0}")]
    ValidationFailed(String),

    /// Invalid license key.
    #[error("Invalid license key")]
    InvalidKey,

    /// License not found.
    #[error("No valid license found")]
    NotFound,
}

/// Result type for license operations.
pub type LicenseResult<T> = std::result::Result<T, LicenseError>;

/// License validator trait.
///
/// Implement this trait to provide custom license validation logic.
pub trait LicenseValidator: Send + Sync + fmt::Debug {
    /// Validate access to a domain.
    fn validate_domain(&self, domain: Domain) -> LicenseResult<()>;

    /// Validate access to a specific feature.
    fn validate_feature(&self, feature: &str) -> LicenseResult<()>;

    /// Check if GPU-native kernels are licensed.
    fn gpu_native_enabled(&self) -> bool;

    /// Get all licensed domains.
    fn licensed_domains(&self) -> &[Domain];

    /// Get license expiry information.
    fn expires_at(&self) -> Option<DateTime<Utc>>;

    /// Check if the license is currently valid.
    fn is_valid(&self) -> bool {
        if let Some(expiry) = self.expires_at() {
            Utc::now() < expiry
        } else {
            true // No expiry = always valid
        }
    }

    /// Get the license tier.
    fn tier(&self) -> LicenseTier;

    /// Get the maximum number of concurrent kernels allowed.
    fn max_kernels(&self) -> Option<usize>;
}

/// License tier levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LicenseTier {
    /// Development/trial license - all features enabled, no production use.
    Development,
    /// Community license - limited features, limited domains.
    Community,
    /// Professional license - most features, most domains.
    Professional,
    /// Enterprise license - all features, all domains, GPU-native.
    Enterprise,
}

impl LicenseTier {
    /// Returns true if this tier supports GPU-native kernels.
    #[must_use]
    pub const fn supports_gpu_native(&self) -> bool {
        matches!(self, LicenseTier::Development | LicenseTier::Enterprise)
    }

    /// Returns the default max kernels for this tier.
    #[must_use]
    pub const fn default_max_kernels(&self) -> Option<usize> {
        match self {
            LicenseTier::Development => None, // Unlimited in dev
            LicenseTier::Community => Some(5),
            LicenseTier::Professional => Some(50),
            LicenseTier::Enterprise => None, // Unlimited
        }
    }
}

/// License identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LicenseId(pub String);

impl LicenseId {
    /// Create a new license ID.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for LicenseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// License configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    /// License identifier.
    pub id: LicenseId,

    /// License tier.
    pub tier: LicenseTier,

    /// Licensed domains.
    pub domains: HashSet<Domain>,

    /// Licensed features (fine-grained).
    pub features: HashSet<String>,

    /// Whether GPU-native kernels are enabled.
    pub gpu_native: bool,

    /// License expiry date (None = never expires).
    pub expires_at: Option<DateTime<Utc>>,

    /// Maximum concurrent kernels (None = unlimited).
    pub max_kernels: Option<usize>,

    /// License holder name.
    pub holder: String,
}

impl License {
    /// Create a development license with all features enabled.
    #[must_use]
    pub fn development() -> Self {
        Self {
            id: LicenseId::new("dev-license"),
            tier: LicenseTier::Development,
            domains: Domain::ALL.iter().copied().collect(),
            features: HashSet::new(), // All features in dev mode
            gpu_native: true,
            expires_at: None,
            max_kernels: None,
            holder: "Development".to_string(),
        }
    }

    /// Create an enterprise license.
    #[must_use]
    pub fn enterprise(holder: impl Into<String>, expires_at: Option<DateTime<Utc>>) -> Self {
        Self {
            id: LicenseId::new(format!("enterprise-{}", chrono::Utc::now().timestamp())),
            tier: LicenseTier::Enterprise,
            domains: Domain::ALL.iter().copied().collect(),
            features: HashSet::new(), // All features in enterprise
            gpu_native: true,
            expires_at,
            max_kernels: None,
            holder: holder.into(),
        }
    }

    /// Create a professional license.
    #[must_use]
    pub fn professional(
        holder: impl Into<String>,
        domains: HashSet<Domain>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            id: LicenseId::new(format!("professional-{}", chrono::Utc::now().timestamp())),
            tier: LicenseTier::Professional,
            domains,
            features: HashSet::new(),
            gpu_native: false, // Professional doesn't include GPU-native
            expires_at,
            max_kernels: Some(50),
            holder: holder.into(),
        }
    }

    /// Create a community license.
    #[must_use]
    pub fn community(holder: impl Into<String>) -> Self {
        let mut domains = HashSet::new();
        domains.insert(Domain::Core);
        domains.insert(Domain::GraphAnalytics);
        domains.insert(Domain::StatisticalML);

        Self {
            id: LicenseId::new(format!("community-{}", chrono::Utc::now().timestamp())),
            tier: LicenseTier::Community,
            domains,
            features: HashSet::new(),
            gpu_native: false,
            expires_at: None, // Community doesn't expire
            max_kernels: Some(5),
            holder: holder.into(),
        }
    }

    /// Add a domain to the license.
    #[must_use]
    pub fn with_domain(mut self, domain: Domain) -> Self {
        self.domains.insert(domain);
        self
    }

    /// Add a feature to the license.
    #[must_use]
    pub fn with_feature(mut self, feature: impl Into<String>) -> Self {
        self.features.insert(feature.into());
        self
    }
}

/// Standard license validator implementation.
#[derive(Debug)]
pub struct StandardLicenseValidator {
    license: License,
}

impl StandardLicenseValidator {
    /// Create a new validator with the given license.
    #[must_use]
    pub fn new(license: License) -> Self {
        Self { license }
    }

    /// Get the underlying license.
    #[must_use]
    pub fn license(&self) -> &License {
        &self.license
    }
}

impl LicenseValidator for StandardLicenseValidator {
    fn validate_domain(&self, domain: Domain) -> LicenseResult<()> {
        // Check expiry first
        if !self.is_valid() {
            return Err(LicenseError::Expired(
                self.license.expires_at.unwrap_or_else(Utc::now),
            ));
        }

        // Development license allows all domains
        if self.license.tier == LicenseTier::Development {
            return Ok(());
        }

        // Enterprise license allows all domains
        if self.license.tier == LicenseTier::Enterprise {
            return Ok(());
        }

        // Check if domain is licensed
        if self.license.domains.contains(&domain) {
            Ok(())
        } else {
            Err(LicenseError::DomainNotLicensed(domain))
        }
    }

    fn validate_feature(&self, feature: &str) -> LicenseResult<()> {
        // Check expiry first
        if !self.is_valid() {
            return Err(LicenseError::Expired(
                self.license.expires_at.unwrap_or_else(Utc::now),
            ));
        }

        // Development and Enterprise allow all features
        if matches!(
            self.license.tier,
            LicenseTier::Development | LicenseTier::Enterprise
        ) {
            return Ok(());
        }

        // Check if feature is explicitly licensed
        if self.license.features.contains(feature) {
            return Ok(());
        }

        // Check if the domain for this feature is licensed
        // Feature format: "Domain.FeatureName"
        if let Some((domain_str, _)) = feature.split_once('.') {
            if let Some(domain) = Domain::from_str(domain_str) {
                if self.license.domains.contains(&domain) {
                    return Ok(());
                }
            }
        }

        Err(LicenseError::FeatureNotLicensed(feature.to_string()))
    }

    fn gpu_native_enabled(&self) -> bool {
        self.license.gpu_native && self.license.tier.supports_gpu_native()
    }

    fn licensed_domains(&self) -> &[Domain] {
        // Return a static slice for development/enterprise
        if matches!(
            self.license.tier,
            LicenseTier::Development | LicenseTier::Enterprise
        ) {
            Domain::ALL
        } else {
            // For other tiers, we can't return a reference to the HashSet
            // This is a limitation - callers should use validate_domain instead
            Domain::ALL // Temporary - proper implementation would use a Vec
        }
    }

    fn expires_at(&self) -> Option<DateTime<Utc>> {
        self.license.expires_at
    }

    fn tier(&self) -> LicenseTier {
        self.license.tier
    }

    fn max_kernels(&self) -> Option<usize> {
        self.license.max_kernels
    }
}

/// Development license that allows all domains (no validation).
///
/// Use this for local development and testing.
#[derive(Debug, Default, Clone)]
pub struct DevelopmentLicense;

impl LicenseValidator for DevelopmentLicense {
    fn validate_domain(&self, _domain: Domain) -> LicenseResult<()> {
        Ok(()) // Always passes in dev mode
    }

    fn validate_feature(&self, _feature: &str) -> LicenseResult<()> {
        Ok(()) // Always passes in dev mode
    }

    fn gpu_native_enabled(&self) -> bool {
        true
    }

    fn licensed_domains(&self) -> &[Domain] {
        Domain::ALL
    }

    fn expires_at(&self) -> Option<DateTime<Utc>> {
        None // Never expires
    }

    fn tier(&self) -> LicenseTier {
        LicenseTier::Development
    }

    fn max_kernels(&self) -> Option<usize> {
        None // Unlimited
    }
}

/// License guard for runtime validation.
///
/// Use this to protect kernel operations with license checks.
#[derive(Debug)]
pub struct LicenseGuard<'a> {
    validator: &'a dyn LicenseValidator,
    domain: Domain,
}

impl<'a> LicenseGuard<'a> {
    /// Create a new license guard.
    #[must_use]
    pub fn new(validator: &'a dyn LicenseValidator, domain: Domain) -> Self {
        Self { validator, domain }
    }

    /// Check if the operation is allowed.
    pub fn check(&self) -> LicenseResult<()> {
        self.validator.validate_domain(self.domain)
    }

    /// Check if a specific feature is allowed.
    pub fn check_feature(&self, feature: &str) -> LicenseResult<()> {
        self.validator.validate_feature(feature)
    }

    /// Check if GPU-native execution is allowed.
    pub fn check_gpu_native(&self) -> LicenseResult<()> {
        if self.validator.gpu_native_enabled() {
            Ok(())
        } else {
            Err(LicenseError::GpuNativeNotLicensed)
        }
    }
}

/// Shared license validator reference.
pub type SharedLicenseValidator = Arc<dyn LicenseValidator>;

/// Create a development license validator.
#[must_use]
pub fn dev_license() -> SharedLicenseValidator {
    Arc::new(DevelopmentLicense)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_development_license() {
        let license = DevelopmentLicense;

        assert!(license.validate_domain(Domain::GraphAnalytics).is_ok());
        assert!(license.validate_domain(Domain::RiskAnalytics).is_ok());
        assert!(license.validate_feature("GraphAnalytics.PageRank").is_ok());
        assert!(license.gpu_native_enabled());
        assert!(license.is_valid());
        assert_eq!(license.tier(), LicenseTier::Development);
    }

    #[test]
    fn test_community_license() {
        let license = License::community("Test User");
        let validator = StandardLicenseValidator::new(license);

        // Community includes Core, GraphAnalytics, StatisticalML
        assert!(validator.validate_domain(Domain::Core).is_ok());
        assert!(validator.validate_domain(Domain::GraphAnalytics).is_ok());
        assert!(validator.validate_domain(Domain::StatisticalML).is_ok());

        // Community does not include RiskAnalytics
        assert!(validator.validate_domain(Domain::RiskAnalytics).is_err());

        // Community does not support GPU-native
        assert!(!validator.gpu_native_enabled());

        // Community has max 5 kernels
        assert_eq!(validator.max_kernels(), Some(5));
    }

    #[test]
    fn test_enterprise_license() {
        let license = License::enterprise("Enterprise User", None);
        let validator = StandardLicenseValidator::new(license);

        // Enterprise includes all domains
        assert!(validator.validate_domain(Domain::GraphAnalytics).is_ok());
        assert!(validator.validate_domain(Domain::RiskAnalytics).is_ok());
        assert!(validator.validate_domain(Domain::Banking).is_ok());

        // Enterprise supports GPU-native
        assert!(validator.gpu_native_enabled());

        // Enterprise has unlimited kernels
        assert_eq!(validator.max_kernels(), None);
    }

    #[test]
    fn test_expired_license() {
        let mut license = License::enterprise("Expired User", None);
        license.expires_at = Some(Utc::now() - chrono::Duration::days(1));

        let validator = StandardLicenseValidator::new(license);

        assert!(!validator.is_valid());
        assert!(validator.validate_domain(Domain::Core).is_err());
    }

    #[test]
    fn test_license_guard() {
        let validator = DevelopmentLicense;
        let guard = LicenseGuard::new(&validator, Domain::GraphAnalytics);

        assert!(guard.check().is_ok());
        assert!(guard.check_feature("GraphAnalytics.PageRank").is_ok());
        assert!(guard.check_gpu_native().is_ok());
    }

    #[test]
    fn test_license_tier_properties() {
        assert!(LicenseTier::Development.supports_gpu_native());
        assert!(LicenseTier::Enterprise.supports_gpu_native());
        assert!(!LicenseTier::Professional.supports_gpu_native());
        assert!(!LicenseTier::Community.supports_gpu_native());
    }
}
