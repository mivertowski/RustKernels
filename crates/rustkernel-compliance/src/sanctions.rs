//! Sanctions screening kernels.
//!
//! This module provides sanctions and PEP screening:
//! - OFAC/UN/EU sanctions list screening
//! - Politically Exposed Persons (PEP) screening

use crate::messages::{
    PEPScreeningInput, PEPScreeningOutput, SanctionsScreeningInput, SanctionsScreeningOutput,
};
use crate::types::{
    PEPEntry, PEPMatch, PEPResult, SanctionsEntry, SanctionsMatch, SanctionsResult,
};
use async_trait::async_trait;
use rustkernel_core::error::Result;
use rustkernel_core::traits::BatchKernel;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::time::Instant;

// ============================================================================
// Sanctions Screening Kernel
// ============================================================================

/// Sanctions list screening kernel.
///
/// Screens names against OFAC, UN, EU and other sanctions lists
/// using fuzzy matching.
#[derive(Debug, Clone)]
pub struct SanctionsScreening {
    metadata: KernelMetadata,
}

impl Default for SanctionsScreening {
    fn default() -> Self {
        Self::new()
    }
}

impl SanctionsScreening {
    /// Create a new sanctions screening kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("compliance/sanctions-screening", Domain::Compliance)
                .with_description("OFAC/UN/EU sanctions list screening")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }

    /// Screen a name against sanctions lists.
    ///
    /// # Arguments
    /// * `name` - Name to screen
    /// * `sanctions_list` - List of sanctions entries
    /// * `min_score` - Minimum match score threshold (0-1)
    /// * `max_matches` - Maximum number of matches to return
    pub fn compute(
        name: &str,
        sanctions_list: &[SanctionsEntry],
        min_score: f64,
        max_matches: usize,
    ) -> SanctionsResult {
        if name.is_empty() || sanctions_list.is_empty() {
            return SanctionsResult {
                query_name: name.to_string(),
                matches: Vec::new(),
                is_hit: false,
            };
        }

        let mut matches: Vec<SanctionsMatch> = sanctions_list
            .iter()
            .filter_map(|entry| {
                let score = Self::match_score(name, entry);
                if score >= min_score {
                    let matched_name = Self::best_matching_name(name, entry);
                    Some(SanctionsMatch {
                        entry_id: entry.id,
                        score,
                        matched_name,
                        source: entry.source.clone(),
                        reason: format!("Name match score: {:.2}%", score * 100.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending
        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(max_matches);

        let is_hit = matches.iter().any(|m| m.score >= 0.85);

        SanctionsResult {
            query_name: name.to_string(),
            matches,
            is_hit,
        }
    }

    /// Batch screen multiple names.
    pub fn compute_batch(
        names: &[String],
        sanctions_list: &[SanctionsEntry],
        min_score: f64,
        max_matches: usize,
    ) -> Vec<SanctionsResult> {
        names
            .iter()
            .map(|name| Self::compute(name, sanctions_list, min_score, max_matches))
            .collect()
    }

    /// Calculate match score between a name and a sanctions entry.
    fn match_score(query: &str, entry: &SanctionsEntry) -> f64 {
        // Check primary name
        let mut best_score = Self::name_similarity(query, &entry.name);

        // Check aliases
        for alias in &entry.aliases {
            let alias_score = Self::name_similarity(query, alias);
            best_score = best_score.max(alias_score);
        }

        best_score
    }

    /// Get the best matching name from an entry.
    fn best_matching_name(query: &str, entry: &SanctionsEntry) -> String {
        let mut best_name = entry.name.clone();
        let mut best_score = Self::name_similarity(query, &entry.name);

        for alias in &entry.aliases {
            let score = Self::name_similarity(query, alias);
            if score > best_score {
                best_score = score;
                best_name = alias.clone();
            }
        }

        best_name
    }

    /// Name similarity using Jaro-Winkler.
    fn name_similarity(s1: &str, s2: &str) -> f64 {
        let s1 = s1.to_lowercase();
        let s2 = s2.to_lowercase();

        if s1 == s2 {
            return 1.0;
        }

        if s1.is_empty() || s2.is_empty() {
            return 0.0;
        }

        // Use the same Jaro-Winkler as EntityResolution
        jaro_winkler(&s1, &s2)
    }
}

impl GpuKernel for SanctionsScreening {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<SanctionsScreeningInput, SanctionsScreeningOutput> for SanctionsScreening {
    async fn execute(&self, input: SanctionsScreeningInput) -> Result<SanctionsScreeningOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.name, &input.sanctions_list, input.min_score, input.max_matches);
        Ok(SanctionsScreeningOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// PEP Screening Kernel
// ============================================================================

/// PEP (Politically Exposed Persons) screening kernel.
///
/// Screens names against PEP lists to identify individuals
/// with political connections.
#[derive(Debug, Clone)]
pub struct PEPScreening {
    metadata: KernelMetadata,
}

impl Default for PEPScreening {
    fn default() -> Self {
        Self::new()
    }
}

impl PEPScreening {
    /// Create a new PEP screening kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("compliance/pep-screening", Domain::Compliance)
                .with_description("Politically Exposed Persons screening")
                .with_throughput(100_000)
                .with_latency_us(10.0),
        }
    }

    /// Screen a name against PEP lists.
    ///
    /// # Arguments
    /// * `name` - Name to screen
    /// * `pep_list` - List of PEP entries
    /// * `min_score` - Minimum match score threshold
    /// * `max_matches` - Maximum number of matches
    pub fn compute(
        name: &str,
        pep_list: &[PEPEntry],
        min_score: f64,
        max_matches: usize,
    ) -> PEPResult {
        if name.is_empty() || pep_list.is_empty() {
            return PEPResult {
                query_name: name.to_string(),
                matches: Vec::new(),
                is_pep: false,
            };
        }

        let mut matches: Vec<PEPMatch> = pep_list
            .iter()
            .filter_map(|entry| {
                let score = jaro_winkler(&name.to_lowercase(), &entry.name.to_lowercase());
                if score >= min_score {
                    Some(PEPMatch {
                        entry_id: entry.id,
                        score,
                        name: entry.name.clone(),
                        position: entry.position.clone(),
                        country: entry.country.clone(),
                        level: entry.level,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending, then by level (higher risk first)
        matches.sort_by(|a, b| {
            match b.score.partial_cmp(&a.score) {
                Some(std::cmp::Ordering::Equal) => a.level.cmp(&b.level),
                other => other.unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        matches.truncate(max_matches);

        let is_pep = matches.iter().any(|m| m.score >= 0.85);

        PEPResult {
            query_name: name.to_string(),
            matches,
            is_pep,
        }
    }

    /// Batch screen multiple names.
    pub fn compute_batch(
        names: &[String],
        pep_list: &[PEPEntry],
        min_score: f64,
        max_matches: usize,
    ) -> Vec<PEPResult> {
        names
            .iter()
            .map(|name| Self::compute(name, pep_list, min_score, max_matches))
            .collect()
    }
}

impl GpuKernel for PEPScreening {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<PEPScreeningInput, PEPScreeningOutput> for PEPScreening {
    async fn execute(&self, input: PEPScreeningInput) -> Result<PEPScreeningOutput> {
        let start = Instant::now();
        let result = Self::compute(&input.name, &input.pep_list, input.min_score, input.max_matches);
        Ok(PEPScreeningOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// String Similarity Helper
// ============================================================================

/// Jaro-Winkler similarity function.
fn jaro_winkler(s1: &str, s2: &str) -> f64 {
    let jaro = jaro(s1, s2);
    let prefix_len = s1
        .chars()
        .zip(s2.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();
    jaro + (prefix_len as f64 * 0.1 * (1.0 - jaro))
}

/// Jaro similarity function.
fn jaro(s1: &str, s2: &str) -> f64 {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    if s1 == s2 {
        return 1.0;
    }

    let match_distance = (len1.max(len2) / 2).saturating_sub(1);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];

    let mut matches = 0usize;
    let mut transpositions = 0usize;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sanctions_list() -> Vec<SanctionsEntry> {
        vec![
            SanctionsEntry {
                id: 1,
                name: "John Doe".to_string(),
                aliases: vec!["Johnny Doe".to_string(), "J. Doe".to_string()],
                source: "OFAC".to_string(),
                program: "SDN".to_string(),
                country: Some("IR".to_string()),
                dob: Some(19700115),
            },
            SanctionsEntry {
                id: 2,
                name: "Evil Corp LLC".to_string(),
                aliases: vec!["Evil Corporation".to_string()],
                source: "OFAC".to_string(),
                program: "SDN".to_string(),
                country: Some("RU".to_string()),
                dob: None,
            },
        ]
    }

    fn create_pep_list() -> Vec<PEPEntry> {
        vec![
            PEPEntry {
                id: 1,
                name: "Vladimir Putin".to_string(),
                position: "President".to_string(),
                country: "RU".to_string(),
                level: 1,
                active: true,
            },
            PEPEntry {
                id: 2,
                name: "Joe Biden".to_string(),
                position: "President".to_string(),
                country: "US".to_string(),
                level: 1,
                active: true,
            },
        ]
    }

    #[test]
    fn test_sanctions_screening_metadata() {
        let kernel = SanctionsScreening::new();
        assert_eq!(kernel.metadata().id, "compliance/sanctions-screening");
        assert_eq!(kernel.metadata().domain, Domain::Compliance);
    }

    #[test]
    fn test_sanctions_exact_match() {
        let list = create_sanctions_list();
        let result = SanctionsScreening::compute("John Doe", &list, 0.5, 10);

        assert!(result.is_hit);
        assert!(!result.matches.is_empty());
        assert_eq!(result.matches[0].entry_id, 1);
        assert!(result.matches[0].score > 0.9);
    }

    #[test]
    fn test_sanctions_alias_match() {
        let list = create_sanctions_list();
        let result = SanctionsScreening::compute("Johnny Doe", &list, 0.5, 10);

        assert!(result.is_hit);
        assert!(!result.matches.is_empty());
        assert!(result.matches[0].score > 0.9);
    }

    #[test]
    fn test_sanctions_fuzzy_match() {
        let list = create_sanctions_list();
        let result = SanctionsScreening::compute("Jon Doe", &list, 0.5, 10);

        assert!(!result.matches.is_empty());
        assert!(result.matches[0].score > 0.7);
    }

    #[test]
    fn test_sanctions_no_match() {
        let list = create_sanctions_list();
        let result = SanctionsScreening::compute("Alice Wonderland", &list, 0.8, 10);

        assert!(!result.is_hit);
        // May have weak matches below threshold
    }

    #[test]
    fn test_pep_screening_metadata() {
        let kernel = PEPScreening::new();
        assert_eq!(kernel.metadata().id, "compliance/pep-screening");
    }

    #[test]
    fn test_pep_exact_match() {
        let list = create_pep_list();
        let result = PEPScreening::compute("Vladimir Putin", &list, 0.5, 10);

        assert!(result.is_pep);
        assert!(!result.matches.is_empty());
        assert_eq!(result.matches[0].level, 1);
    }

    #[test]
    fn test_pep_fuzzy_match() {
        let list = create_pep_list();
        let result = PEPScreening::compute("Vladmir Putin", &list, 0.7, 10);

        // Should still match with fuzzy matching
        assert!(!result.matches.is_empty());
        assert!(result.matches[0].score > 0.8);
    }

    #[test]
    fn test_empty_inputs() {
        let list = create_sanctions_list();

        let result1 = SanctionsScreening::compute("", &list, 0.5, 10);
        assert!(result1.matches.is_empty());

        let result2 = SanctionsScreening::compute("John Doe", &[], 0.5, 10);
        assert!(result2.matches.is_empty());
    }
}
