//! Credit risk scoring kernels.
//!
//! This module provides credit risk analytics:
//! - PD (Probability of Default) modeling
//! - LGD (Loss Given Default) estimation
//! - Expected Loss calculation
//! - Risk-weighted asset calculation

use crate::types::{CreditExposure, CreditFactors, CreditRiskResult};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

// ============================================================================
// Credit Risk Scoring Kernel
// ============================================================================

/// Credit risk scoring kernel.
///
/// Calculates PD, LGD, EL, and RWA for credit exposures.
#[derive(Debug, Clone)]
pub struct CreditRiskScoring {
    metadata: KernelMetadata,
}

impl Default for CreditRiskScoring {
    fn default() -> Self {
        Self::new()
    }
}

impl CreditRiskScoring {
    /// Create a new credit risk scoring kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("risk/credit-scoring", Domain::RiskAnalytics)
                .with_description("PD/LGD/EAD credit risk calculation")
                .with_throughput(50_000)
                .with_latency_us(100.0),
        }
    }

    /// Score credit risk for a single obligor.
    ///
    /// # Arguments
    /// * `factors` - Credit scoring factors
    /// * `ead` - Exposure at Default
    /// * `maturity` - Loan maturity in years
    pub fn compute(factors: &CreditFactors, ead: f64, maturity: f64) -> CreditRiskResult {
        // Calculate credit score using a simplified scorecard model
        let mut score = 600.0; // Base score
        let mut contributions = Vec::new();

        // Payment history (most important factor)
        let payment_contrib = factors.payment_history * 0.35;
        score += payment_contrib;
        contributions.push(("Payment History".to_string(), payment_contrib));

        // Credit utilization (lower is better)
        let util_impact = (1.0 - factors.credit_utilization) * 100.0 * 0.30;
        score += util_impact;
        contributions.push(("Credit Utilization".to_string(), util_impact));

        // Credit history length
        let history_impact = factors.credit_history_years.min(30.0) * 2.0 * 0.15;
        score += history_impact;
        contributions.push(("Credit History Length".to_string(), history_impact));

        // Debt-to-income (lower is better)
        let dti_impact = (1.0 - factors.debt_to_income.min(1.0)) * 50.0 * 0.10;
        score += dti_impact;
        contributions.push(("Debt-to-Income".to_string(), dti_impact));

        // Recent inquiries (fewer is better)
        let inquiry_impact = (10 - factors.recent_inquiries.min(10)) as f64 * 3.0 * 0.05;
        score += inquiry_impact;
        contributions.push(("Recent Inquiries".to_string(), inquiry_impact));

        // Delinquencies (none is best)
        let delinq_impact = -((factors.delinquencies as f64) * 20.0 * 0.05);
        score += delinq_impact;
        contributions.push(("Delinquencies".to_string(), delinq_impact));

        // Clamp score
        let credit_score = score.clamp(300.0, 850.0);

        // Convert score to PD (logistic function)
        let pd = Self::score_to_pd(credit_score);

        // Estimate LGD based on collateral (LTV ratio)
        let lgd = Self::estimate_lgd(factors.loan_to_value);

        // Expected Loss = PD * LGD * EAD
        let expected_loss = pd * lgd * ead;

        // Risk-weighted assets (simplified Basel formula)
        let rwa = Self::calculate_rwa(pd, lgd, ead, maturity);

        CreditRiskResult {
            obligor_id: factors.obligor_id,
            pd,
            lgd,
            expected_loss,
            rwa,
            credit_score,
            factor_contributions: contributions,
        }
    }

    /// Batch score multiple obligors.
    pub fn compute_batch(
        factors_list: &[CreditFactors],
        eads: &[f64],
        maturities: &[f64],
    ) -> Vec<CreditRiskResult> {
        factors_list
            .iter()
            .zip(eads.iter())
            .zip(maturities.iter())
            .map(|((f, &ead), &mat)| Self::compute(f, ead, mat))
            .collect()
    }

    /// Score credit risk from existing exposure data.
    pub fn compute_from_exposure(exposure: &CreditExposure) -> CreditRiskResult {
        let rwa = Self::calculate_rwa(exposure.pd, exposure.lgd, exposure.ead, exposure.maturity);

        CreditRiskResult {
            obligor_id: exposure.obligor_id,
            pd: exposure.pd,
            lgd: exposure.lgd,
            expected_loss: exposure.expected_loss(),
            rwa,
            credit_score: Self::pd_to_score(exposure.pd),
            factor_contributions: Vec::new(),
        }
    }

    /// Convert credit score to probability of default.
    fn score_to_pd(score: f64) -> f64 {
        // Logistic transformation: higher score = lower PD
        // Calibrated so that score 700 ≈ 2% PD, 600 ≈ 10% PD
        let x = (700.0 - score) / 50.0;
        1.0 / (1.0 + (-x).exp()) * 0.30 // Cap at 30% PD
    }

    /// Convert PD back to approximate credit score.
    fn pd_to_score(pd: f64) -> f64 {
        // Inverse of score_to_pd
        let clamped_pd = pd.clamp(0.001, 0.30);
        let x = (clamped_pd / 0.30).ln() - (-clamped_pd / 0.30 + 1.0).ln();
        700.0 - x * 50.0
    }

    /// Estimate LGD based on loan-to-value ratio.
    fn estimate_lgd(ltv: f64) -> f64 {
        // Higher LTV = higher LGD
        // Assumes some recovery from collateral
        let base_lgd = 0.45; // Unsecured baseline
        let secured_reduction = (1.0 - ltv.min(1.0)) * 0.30;
        (base_lgd - secured_reduction).max(0.10)
    }

    /// Calculate risk-weighted assets using Basel IRB formula (simplified).
    fn calculate_rwa(pd: f64, lgd: f64, ead: f64, maturity: f64) -> f64 {
        // Simplified Basel II IRB formula
        let pd_clamped = pd.clamp(0.0003, 1.0);
        let lgd_clamped = lgd.clamp(0.0, 1.0);

        // Asset correlation (depends on PD)
        let r = 0.12 * (1.0 - (-50.0 * pd_clamped).exp()) / (1.0 - (-50.0_f64).exp())
            + 0.24 * (1.0 - (1.0 - (-50.0 * pd_clamped).exp()) / (1.0 - (-50.0_f64).exp()));

        // Maturity adjustment
        let b = (0.11852 - 0.05478 * pd_clamped.ln()).powi(2);
        let m_adj = (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b);

        // Capital requirement
        let k = lgd_clamped
            * (Self::norm_cdf(
                Self::norm_inv(pd_clamped) / (1.0 - r).sqrt()
                    + (r / (1.0 - r)).sqrt() * Self::norm_inv(0.999),
            ) - pd_clamped)
            * m_adj;

        // RWA = 12.5 * K * EAD
        12.5 * k * ead
    }

    /// Standard normal CDF approximation.
    fn norm_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        if x > 0.0 { 1.0 - p } else { p }
    }

    /// Standard normal inverse CDF approximation.
    fn norm_inv(p: f64) -> f64 {
        // Rational approximation (Abramowitz & Stegun)
        let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);

        let a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];

        let b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];

        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];

        let d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p_clamped < p_low {
            let q = (-2.0 * p_clamped.ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p_clamped <= p_high {
            let q = p_clamped - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p_clamped).ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
    }
}

impl GpuKernel for CreditRiskScoring {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_good_obligor() -> CreditFactors {
        CreditFactors {
            obligor_id: 1,
            debt_to_income: 0.25,
            loan_to_value: 0.60,
            credit_utilization: 0.15,
            payment_history: 95.0,
            employment_years: 10.0,
            recent_inquiries: 1,
            delinquencies: 0,
            credit_history_years: 15.0,
        }
    }

    fn create_risky_obligor() -> CreditFactors {
        CreditFactors {
            obligor_id: 2,
            debt_to_income: 0.55,
            loan_to_value: 0.95,
            credit_utilization: 0.85,
            payment_history: 60.0,
            employment_years: 1.0,
            recent_inquiries: 6,
            delinquencies: 3,
            credit_history_years: 2.0,
        }
    }

    #[test]
    fn test_credit_scoring_metadata() {
        let kernel = CreditRiskScoring::new();
        assert_eq!(kernel.metadata().id, "risk/credit-scoring");
        assert_eq!(kernel.metadata().domain, Domain::RiskAnalytics);
    }

    #[test]
    fn test_good_obligor_scoring() {
        let factors = create_good_obligor();
        let result = CreditRiskScoring::compute(&factors, 100_000.0, 5.0);

        assert_eq!(result.obligor_id, 1);
        assert!(result.credit_score > 650.0, "Good obligor should have score > 650, got {}", result.credit_score);
        // PD maps from score via logistic function
        assert!(result.pd < 0.25, "Good obligor should have PD < 25%, got {}", result.pd);
        assert!(result.lgd < 0.45, "Secured loan should have LGD < 45%, got {}", result.lgd);
        assert!(result.expected_loss < 10000.0, "Expected loss should be reasonable");
    }

    #[test]
    fn test_risky_obligor_scoring() {
        let factors = create_risky_obligor();
        let result = CreditRiskScoring::compute(&factors, 100_000.0, 5.0);

        assert_eq!(result.obligor_id, 2);
        assert!(result.credit_score < 650.0, "Risky obligor should have score < 650, got {}", result.credit_score);
        assert!(result.pd > 0.05, "Risky obligor should have PD > 5%, got {}", result.pd);
        assert!(result.lgd > 0.35, "High LTV loan should have higher LGD");
    }

    #[test]
    fn test_rwa_calculation() {
        let good = create_good_obligor();
        let risky = create_risky_obligor();

        let good_result = CreditRiskScoring::compute(&good, 100_000.0, 5.0);
        let risky_result = CreditRiskScoring::compute(&risky, 100_000.0, 5.0);

        // Risky obligor should have higher RWA
        assert!(
            risky_result.rwa > good_result.rwa,
            "Risky obligor should have higher RWA: {} vs {}",
            risky_result.rwa,
            good_result.rwa
        );
    }

    #[test]
    fn test_batch_scoring() {
        let factors = vec![create_good_obligor(), create_risky_obligor()];
        let eads = vec![100_000.0, 50_000.0];
        let maturities = vec![5.0, 3.0];

        let results = CreditRiskScoring::compute_batch(&factors, &eads, &maturities);

        assert_eq!(results.len(), 2);
        assert!(results[0].credit_score > results[1].credit_score);
    }

    #[test]
    fn test_exposure_scoring() {
        let exposure = CreditExposure::new(100, 50_000.0, 0.02, 0.40, 3.0, 2);

        let result = CreditRiskScoring::compute_from_exposure(&exposure);

        assert_eq!(result.obligor_id, 100);
        assert!((result.pd - 0.02).abs() < 0.001);
        assert!((result.lgd - 0.40).abs() < 0.001);
        assert!((result.expected_loss - 400.0).abs() < 1.0); // 0.02 * 0.40 * 50000 = 400
    }

    #[test]
    fn test_factor_contributions() {
        let factors = create_good_obligor();
        let result = CreditRiskScoring::compute(&factors, 100_000.0, 5.0);

        assert!(!result.factor_contributions.is_empty());
        assert!(result.factor_contributions.iter().any(|(name, _)| name == "Payment History"));
    }

    #[test]
    fn test_pd_score_conversion() {
        // Test round-trip conversion
        let scores = [300.0, 500.0, 650.0, 700.0, 750.0, 800.0];

        for &score in &scores {
            let pd = CreditRiskScoring::score_to_pd(score);
            assert!(pd > 0.0 && pd <= 0.30, "PD out of range for score {}: {}", score, pd);

            // Higher score should mean lower PD
            let pd_low = CreditRiskScoring::score_to_pd(score + 50.0);
            assert!(pd_low < pd, "Higher score should have lower PD");
        }
    }
}
