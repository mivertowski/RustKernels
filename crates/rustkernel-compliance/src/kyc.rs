//! Know Your Customer (KYC) kernels.
//!
//! This module provides KYC analysis:
//! - Risk scoring
//! - Entity resolution/matching

use crate::types::{
    Entity, EntityMatch, EntityResolutionResult, KYCFactors, KYCResult, RiskTier,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};

// ============================================================================
// KYC Scoring Kernel
// ============================================================================

/// KYC risk scoring kernel.
///
/// Aggregates multiple risk factors into an overall KYC risk score.
#[derive(Debug, Clone)]
pub struct KYCScoring {
    metadata: KernelMetadata,
}

impl Default for KYCScoring {
    fn default() -> Self {
        Self::new()
    }
}

impl KYCScoring {
    /// Create a new KYC scoring kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("compliance/kyc-scoring", Domain::Compliance)
                .with_description("KYC risk factor aggregation")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Compute KYC risk score for a customer.
    ///
    /// # Arguments
    /// * `factors` - KYC risk factors
    /// * `weights` - Optional custom weights for each factor
    pub fn compute(factors: &KYCFactors, weights: Option<&KYCWeights>) -> KYCResult {
        let default_weights = KYCWeights::default();
        let w = weights.unwrap_or(&default_weights);

        // Calculate weighted score
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut contributions = Vec::new();

        // Country risk
        let country_contribution = factors.country_risk * w.country;
        weighted_sum += country_contribution;
        total_weight += w.country;
        contributions.push(("Country Risk".to_string(), country_contribution / w.country));

        // Industry risk
        let industry_contribution = factors.industry_risk * w.industry;
        weighted_sum += industry_contribution;
        total_weight += w.industry;
        contributions.push(("Industry Risk".to_string(), industry_contribution / w.industry));

        // Product risk
        let product_contribution = factors.product_risk * w.product;
        weighted_sum += product_contribution;
        total_weight += w.product;
        contributions.push(("Product Risk".to_string(), product_contribution / w.product));

        // Transaction risk
        let tx_contribution = factors.transaction_risk * w.transaction;
        weighted_sum += tx_contribution;
        total_weight += w.transaction;
        contributions.push(("Transaction Risk".to_string(), tx_contribution / w.transaction));

        // Documentation (inverse - higher is better)
        let doc_risk = 100.0 - factors.documentation_score;
        let doc_contribution = doc_risk * w.documentation;
        weighted_sum += doc_contribution;
        total_weight += w.documentation;
        contributions.push(("Documentation Gap".to_string(), doc_contribution / w.documentation));

        // Tenure (inverse - longer is better)
        let tenure_risk = (10.0 - factors.tenure_years.min(10.0)) * 10.0;
        let tenure_contribution = tenure_risk * w.tenure;
        weighted_sum += tenure_contribution;
        total_weight += w.tenure;
        contributions.push(("Tenure Risk".to_string(), tenure_contribution / w.tenure));

        let risk_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        let risk_tier = RiskTier::from(risk_score);

        KYCResult {
            customer_id: factors.customer_id,
            risk_score,
            risk_tier,
            factor_contributions: contributions,
        }
    }

    /// Batch score multiple customers.
    pub fn compute_batch(
        factors_list: &[KYCFactors],
        weights: Option<&KYCWeights>,
    ) -> Vec<KYCResult> {
        factors_list
            .iter()
            .map(|f| Self::compute(f, weights))
            .collect()
    }
}

impl GpuKernel for KYCScoring {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Weights for KYC risk factors.
#[derive(Debug, Clone)]
pub struct KYCWeights {
    /// Country risk weight.
    pub country: f64,
    /// Industry risk weight.
    pub industry: f64,
    /// Product risk weight.
    pub product: f64,
    /// Transaction pattern weight.
    pub transaction: f64,
    /// Documentation weight.
    pub documentation: f64,
    /// Tenure weight.
    pub tenure: f64,
}

impl Default for KYCWeights {
    fn default() -> Self {
        Self {
            country: 0.25,
            industry: 0.20,
            product: 0.15,
            transaction: 0.20,
            documentation: 0.10,
            tenure: 0.10,
        }
    }
}

// ============================================================================
// Entity Resolution Kernel
// ============================================================================

/// Entity resolution (fuzzy matching) kernel.
///
/// Matches entities using fuzzy string matching and other attributes.
#[derive(Debug, Clone)]
pub struct EntityResolution {
    metadata: KernelMetadata,
}

impl Default for EntityResolution {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityResolution {
    /// Create a new entity resolution kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("compliance/entity-resolution", Domain::Compliance)
                .with_description("Fuzzy entity matching")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Match a query entity against a list of candidates.
    ///
    /// # Arguments
    /// * `query` - Entity to match
    /// * `candidates` - List of candidate entities
    /// * `min_score` - Minimum match score threshold
    /// * `max_matches` - Maximum number of matches to return
    pub fn compute(
        query: &Entity,
        candidates: &[Entity],
        min_score: f64,
        max_matches: usize,
    ) -> EntityResolutionResult {
        let mut matches: Vec<EntityMatch> = candidates
            .iter()
            .filter_map(|candidate| {
                let (name_score, date_score, country_match) =
                    Self::compute_scores(query, candidate);

                // Weighted overall score
                let mut score = name_score * 0.6;
                if date_score > 0.0 {
                    score += date_score * 0.25;
                } else {
                    score += 0.125; // Neutral if no date to compare
                }
                if country_match {
                    score += 0.15;
                }

                if score >= min_score {
                    Some(EntityMatch {
                        entity_id: candidate.id,
                        score,
                        name_score,
                        date_score,
                        country_match,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top matches
        matches.truncate(max_matches);

        EntityResolutionResult {
            query_id: query.id,
            matches,
        }
    }

    /// Compute individual scores between two entities.
    fn compute_scores(query: &Entity, candidate: &Entity) -> (f64, f64, bool) {
        // Name similarity (best match across all names/aliases)
        let mut best_name_score = Self::name_similarity(&query.name, &candidate.name);

        for alias in &candidate.aliases {
            let alias_score = Self::name_similarity(&query.name, alias);
            best_name_score = best_name_score.max(alias_score);
        }

        for query_alias in &query.aliases {
            let alias_score = Self::name_similarity(query_alias, &candidate.name);
            best_name_score = best_name_score.max(alias_score);

            for candidate_alias in &candidate.aliases {
                let aa_score = Self::name_similarity(query_alias, candidate_alias);
                best_name_score = best_name_score.max(aa_score);
            }
        }

        // Date similarity
        let date_score = match (query.date, candidate.date) {
            (Some(qd), Some(cd)) => Self::date_similarity(qd, cd),
            _ => 0.0,
        };

        // Country match
        let country_match = match (&query.country, &candidate.country) {
            (Some(qc), Some(cc)) => qc.eq_ignore_ascii_case(cc),
            _ => false,
        };

        (best_name_score, date_score, country_match)
    }

    /// Calculate name similarity using Jaro-Winkler distance.
    fn name_similarity(s1: &str, s2: &str) -> f64 {
        let s1 = s1.to_lowercase();
        let s2 = s2.to_lowercase();

        if s1 == s2 {
            return 1.0;
        }

        if s1.is_empty() || s2.is_empty() {
            return 0.0;
        }

        Self::jaro_winkler(&s1, &s2)
    }

    /// Jaro-Winkler similarity.
    fn jaro_winkler(s1: &str, s2: &str) -> f64 {
        let jaro = Self::jaro(s1, s2);

        // Calculate common prefix length (up to 4 chars)
        let prefix_len = s1
            .chars()
            .zip(s2.chars())
            .take(4)
            .take_while(|(a, b)| a == b)
            .count();

        // Winkler modification
        jaro + (prefix_len as f64 * 0.1 * (1.0 - jaro))
    }

    /// Jaro similarity.
    fn jaro(s1: &str, s2: &str) -> f64 {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let len1 = s1_chars.len();
        let len2 = s2_chars.len();

        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        let match_distance = (len1.max(len2) / 2).saturating_sub(1);

        let mut s1_matches = vec![false; len1];
        let mut s2_matches = vec![false; len2];

        let mut matches = 0usize;
        let mut transpositions = 0usize;

        // Find matches
        for i in 0..len1 {
            let start = i.saturating_sub(match_distance);
            let end = (i + match_distance + 1).min(len2);

            for j in start..end {
                if s2_matches[j] || s1_chars[i] != s2_chars[j] {
                    continue;
                }
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1;
                break;
            }
        }

        if matches == 0 {
            return 0.0;
        }

        // Count transpositions
        let mut k = 0usize;
        for i in 0..len1 {
            if !s1_matches[i] {
                continue;
            }
            while !s2_matches[k] {
                k += 1;
            }
            if s1_chars[i] != s2_chars[k] {
                transpositions += 1;
            }
            k += 1;
        }

        let m = matches as f64;
        let t = transpositions as f64 / 2.0;

        (m / len1 as f64 + m / len2 as f64 + (m - t) / m) / 3.0
    }

    /// Date similarity (YYYYMMDD format).
    fn date_similarity(d1: u32, d2: u32) -> f64 {
        if d1 == d2 {
            return 1.0;
        }

        // Extract year, month, day
        let y1 = d1 / 10000;
        let m1 = (d1 % 10000) / 100;
        let _day1 = d1 % 100;

        let y2 = d2 / 10000;
        let m2 = (d2 % 10000) / 100;
        let _day2 = d2 % 100;

        // Same year and month is close
        if y1 == y2 && m1 == m2 {
            return 0.9;
        }

        // Same year
        if y1 == y2 {
            return 0.7;
        }

        // Within a few years
        let year_diff = (y1 as i32 - y2 as i32).unsigned_abs();
        if year_diff <= 2 {
            return 0.5;
        }
        if year_diff <= 5 {
            return 0.3;
        }

        0.0
    }
}

impl GpuKernel for EntityResolution {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EntityType;

    #[test]
    fn test_kyc_scoring_metadata() {
        let kernel = KYCScoring::new();
        assert_eq!(kernel.metadata().id, "compliance/kyc-scoring");
        assert_eq!(kernel.metadata().domain, Domain::Compliance);
    }

    #[test]
    fn test_kyc_scoring_low_risk() {
        let factors = KYCFactors {
            customer_id: 1,
            country_risk: 10.0,
            industry_risk: 15.0,
            product_risk: 10.0,
            transaction_risk: 5.0,
            documentation_score: 95.0,
            tenure_years: 8.0,
        };

        let result = KYCScoring::compute(&factors, None);

        assert_eq!(result.customer_id, 1);
        assert!(result.risk_score < 25.0);
        assert_eq!(result.risk_tier, RiskTier::Low);
    }

    #[test]
    fn test_kyc_scoring_high_risk() {
        let factors = KYCFactors {
            customer_id: 2,
            country_risk: 80.0,
            industry_risk: 70.0,
            product_risk: 60.0,
            transaction_risk: 75.0,
            documentation_score: 40.0,
            tenure_years: 0.5,
        };

        let result = KYCScoring::compute(&factors, None);

        assert!(result.risk_score > 50.0);
        assert!(matches!(
            result.risk_tier,
            RiskTier::High | RiskTier::VeryHigh
        ));
    }

    #[test]
    fn test_entity_resolution_metadata() {
        let kernel = EntityResolution::new();
        assert_eq!(kernel.metadata().id, "compliance/entity-resolution");
    }

    #[test]
    fn test_entity_resolution_exact_match() {
        let query = Entity {
            id: 1,
            name: "John Smith".to_string(),
            aliases: vec![],
            date: Some(19800115),
            country: Some("US".to_string()),
            entity_type: EntityType::Individual,
        };

        let candidates = vec![
            Entity {
                id: 100,
                name: "John Smith".to_string(),
                aliases: vec![],
                date: Some(19800115),
                country: Some("US".to_string()),
                entity_type: EntityType::Individual,
            },
            Entity {
                id: 101,
                name: "Jane Doe".to_string(),
                aliases: vec![],
                date: Some(19850620),
                country: Some("UK".to_string()),
                entity_type: EntityType::Individual,
            },
        ];

        let result = EntityResolution::compute(&query, &candidates, 0.5, 10);

        assert!(!result.matches.is_empty());
        assert_eq!(result.matches[0].entity_id, 100);
        assert!(result.matches[0].score > 0.9);
    }

    #[test]
    fn test_entity_resolution_fuzzy_match() {
        let query = Entity {
            id: 1,
            name: "Jon Smyth".to_string(), // Misspelled
            aliases: vec![],
            date: None,
            country: Some("US".to_string()),
            entity_type: EntityType::Individual,
        };

        let candidates = vec![Entity {
            id: 100,
            name: "John Smith".to_string(),
            aliases: vec!["Johnny Smith".to_string()],
            date: None,
            country: Some("US".to_string()),
            entity_type: EntityType::Individual,
        }];

        let result = EntityResolution::compute(&query, &candidates, 0.5, 10);

        // Should still match with decent score due to fuzzy matching
        assert!(!result.matches.is_empty());
        assert!(result.matches[0].score > 0.6);
    }

    #[test]
    fn test_entity_resolution_alias_match() {
        let query = Entity {
            id: 1,
            name: "Johnny Smith".to_string(),
            aliases: vec![],
            date: None,
            country: None,
            entity_type: EntityType::Individual,
        };

        let candidates = vec![Entity {
            id: 100,
            name: "John Smith".to_string(),
            aliases: vec!["Johnny Smith".to_string(), "J. Smith".to_string()],
            date: None,
            country: None,
            entity_type: EntityType::Individual,
        }];

        let result = EntityResolution::compute(&query, &candidates, 0.5, 10);

        // Should get perfect match via alias
        assert!(!result.matches.is_empty());
        assert!(result.matches[0].name_score > 0.95);
    }
}
