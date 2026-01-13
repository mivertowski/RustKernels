//! Healthcare analytics kernels.
//!
//! This module provides GPU-accelerated healthcare algorithms:
//! - DrugInteractionPrediction - Multi-drug interaction analysis
//! - ClinicalPathwayConformance - Treatment guideline checking

use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Drug Interaction Prediction Kernel
// ============================================================================

/// Configuration for drug interaction prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInteractionConfig {
    /// Maximum interaction order to check (2 = pairwise, 3 = triplets, etc.).
    pub max_order: usize,
    /// Minimum confidence for reported interactions.
    pub min_confidence: f64,
    /// Include known interactions in output.
    pub include_known: bool,
    /// Severity levels to include.
    pub severity_filter: Vec<Severity>,
}

impl Default for DrugInteractionConfig {
    fn default() -> Self {
        Self {
            max_order: 3,
            min_confidence: 0.5,
            include_known: true,
            severity_filter: vec![Severity::Major, Severity::Moderate, Severity::Minor],
        }
    }
}

/// Drug severity level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Severity {
    /// Life-threatening or major organ damage.
    Major,
    /// Significant but not life-threatening.
    Moderate,
    /// Minor effects, usually manageable.
    Minor,
    /// Theoretical or minimal clinical significance.
    Minimal,
}

/// A drug entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drug {
    /// Drug identifier (e.g., RxNorm CUI).
    pub id: String,
    /// Drug name.
    pub name: String,
    /// Drug class/category.
    pub drug_class: Option<String>,
    /// Mechanism of action features.
    pub moa_features: Vec<f64>,
    /// Target proteins/receptors.
    pub targets: Vec<String>,
}

/// Known interaction entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownInteraction {
    /// Drug IDs involved.
    pub drug_ids: Vec<String>,
    /// Severity of interaction.
    pub severity: Severity,
    /// Description of the interaction.
    pub description: String,
    /// Clinical recommendation.
    pub recommendation: String,
}

/// Drug interaction knowledge base.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InteractionKnowledgeBase {
    /// Known pairwise interactions.
    pub pairwise: HashMap<(String, String), KnownInteraction>,
    /// Known higher-order interactions.
    pub higher_order: HashMap<Vec<String>, KnownInteraction>,
    /// Drug class interactions.
    pub class_interactions: HashMap<(String, String), Severity>,
}

impl InteractionKnowledgeBase {
    /// Check if drugs have a known interaction.
    pub fn get_known_interaction(&self, drug_ids: &[String]) -> Option<&KnownInteraction> {
        if drug_ids.len() == 2 {
            let key = Self::normalize_pair(&drug_ids[0], &drug_ids[1]);
            self.pairwise.get(&key)
        } else {
            let mut sorted = drug_ids.to_vec();
            sorted.sort();
            self.higher_order.get(&sorted)
        }
    }

    fn normalize_pair(a: &str, b: &str) -> (String, String) {
        if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }

    /// Add a known interaction.
    pub fn add_interaction(&mut self, interaction: KnownInteraction) {
        if interaction.drug_ids.len() == 2 {
            let key = Self::normalize_pair(&interaction.drug_ids[0], &interaction.drug_ids[1]);
            self.pairwise.insert(key, interaction);
        } else {
            let mut sorted = interaction.drug_ids.clone();
            sorted.sort();
            self.higher_order.insert(sorted, interaction);
        }
    }
}

/// Predicted drug interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedInteraction {
    /// Drugs involved.
    pub drug_ids: Vec<String>,
    /// Drug names.
    pub drug_names: Vec<String>,
    /// Predicted severity.
    pub severity: Severity,
    /// Confidence score (0-1).
    pub confidence: f64,
    /// Whether this is a known interaction.
    pub is_known: bool,
    /// Interaction mechanism (if predicted).
    pub mechanism: Option<String>,
    /// Risk score.
    pub risk_score: f64,
}

/// Result of drug interaction prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInteractionResult {
    /// All predicted interactions.
    pub interactions: Vec<PredictedInteraction>,
    /// High-risk drug combinations.
    pub high_risk_combinations: Vec<Vec<String>>,
    /// Overall polypharmacy risk score.
    pub polypharmacy_risk: f64,
    /// Recommendations.
    pub recommendations: Vec<String>,
}

/// Drug Interaction Prediction kernel.
///
/// Analyzes drug combinations for potential interactions using
/// mechanism-of-action features and known interaction databases.
/// Supports pairwise and higher-order (multi-drug) interactions.
#[derive(Debug, Clone)]
pub struct DrugInteractionPrediction {
    metadata: KernelMetadata,
}

impl Default for DrugInteractionPrediction {
    fn default() -> Self {
        Self::new()
    }
}

impl DrugInteractionPrediction {
    /// Create a new Drug Interaction Prediction kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch(
                "ml/drug-interaction-prediction",
                Domain::StatisticalML,
            )
            .with_description("Multi-drug interaction prediction using mechanism features")
            .with_throughput(5_000)
            .with_latency_us(200.0),
        }
    }

    /// Predict interactions for a set of drugs.
    pub fn predict(
        drugs: &[Drug],
        knowledge_base: &InteractionKnowledgeBase,
        config: &DrugInteractionConfig,
    ) -> DrugInteractionResult {
        if drugs.is_empty() {
            return DrugInteractionResult {
                interactions: Vec::new(),
                high_risk_combinations: Vec::new(),
                polypharmacy_risk: 0.0,
                recommendations: Vec::new(),
            };
        }

        let mut interactions = Vec::new();
        let mut high_risk = Vec::new();

        // Check pairwise interactions
        for i in 0..drugs.len() {
            for j in (i + 1)..drugs.len() {
                let drug_ids = vec![drugs[i].id.clone(), drugs[j].id.clone()];
                let drug_names = vec![drugs[i].name.clone(), drugs[j].name.clone()];

                // Check known interactions
                if let Some(known) = knowledge_base.get_known_interaction(&drug_ids) {
                    if config.include_known && config.severity_filter.contains(&known.severity) {
                        interactions.push(PredictedInteraction {
                            drug_ids: drug_ids.clone(),
                            drug_names: drug_names.clone(),
                            severity: known.severity,
                            confidence: 1.0,
                            is_known: true,
                            mechanism: Some(known.description.clone()),
                            risk_score: Self::severity_to_risk(known.severity),
                        });

                        if known.severity == Severity::Major {
                            high_risk.push(drug_ids.clone());
                        }
                    }
                } else {
                    // Predict interaction based on features
                    let (severity, confidence) =
                        Self::predict_pairwise(&drugs[i], &drugs[j], knowledge_base);

                    if confidence >= config.min_confidence
                        && config.severity_filter.contains(&severity)
                    {
                        let risk = Self::severity_to_risk(severity) * confidence;

                        interactions.push(PredictedInteraction {
                            drug_ids: drug_ids.clone(),
                            drug_names,
                            severity,
                            confidence,
                            is_known: false,
                            mechanism: Self::predict_mechanism(&drugs[i], &drugs[j]),
                            risk_score: risk,
                        });

                        if severity == Severity::Major && confidence > 0.7 {
                            high_risk.push(drug_ids);
                        }
                    }
                }
            }
        }

        // Check higher-order interactions if configured
        if config.max_order >= 3 && drugs.len() >= 3 {
            for i in 0..drugs.len() {
                for j in (i + 1)..drugs.len() {
                    for k in (j + 1)..drugs.len() {
                        let drug_ids = vec![
                            drugs[i].id.clone(),
                            drugs[j].id.clone(),
                            drugs[k].id.clone(),
                        ];

                        let (severity, confidence) =
                            Self::predict_triplet(&drugs[i], &drugs[j], &drugs[k], knowledge_base);

                        if confidence >= config.min_confidence {
                            interactions.push(PredictedInteraction {
                                drug_ids: drug_ids.clone(),
                                drug_names: vec![
                                    drugs[i].name.clone(),
                                    drugs[j].name.clone(),
                                    drugs[k].name.clone(),
                                ],
                                severity,
                                confidence,
                                is_known: false,
                                mechanism: Some("Complex multi-drug interaction".to_string()),
                                risk_score: Self::severity_to_risk(severity) * confidence,
                            });
                        }
                    }
                }
            }
        }

        // Calculate polypharmacy risk
        let polypharmacy_risk = Self::calculate_polypharmacy_risk(drugs.len(), &interactions);

        // Generate recommendations
        let recommendations = Self::generate_recommendations(&interactions, &high_risk);

        // Sort interactions by risk
        interactions.sort_by(|a, b| {
            b.risk_score
                .partial_cmp(&a.risk_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        DrugInteractionResult {
            interactions,
            high_risk_combinations: high_risk,
            polypharmacy_risk,
            recommendations,
        }
    }

    /// Predict pairwise interaction from drug features.
    fn predict_pairwise(
        drug_a: &Drug,
        drug_b: &Drug,
        kb: &InteractionKnowledgeBase,
    ) -> (Severity, f64) {
        // Check class-level interactions
        if let (Some(class_a), Some(class_b)) = (&drug_a.drug_class, &drug_b.drug_class) {
            let key = if class_a < class_b {
                (class_a.clone(), class_b.clone())
            } else {
                (class_b.clone(), class_a.clone())
            };

            if let Some(&severity) = kb.class_interactions.get(&key) {
                return (severity, 0.8);
            }
        }

        // Compute feature-based similarity
        let moa_similarity = Self::cosine_similarity(&drug_a.moa_features, &drug_b.moa_features);

        // Check target overlap
        let target_overlap = Self::jaccard_similarity(&drug_a.targets, &drug_b.targets);

        // Heuristic: high target overlap + different MOA = higher risk
        let risk_score = target_overlap * (1.0 - moa_similarity) + moa_similarity * 0.3;

        let (severity, confidence) = if risk_score > 0.7 {
            (Severity::Major, risk_score)
        } else if risk_score > 0.5 {
            (Severity::Moderate, risk_score)
        } else if risk_score > 0.3 {
            (Severity::Minor, risk_score)
        } else {
            (Severity::Minimal, risk_score)
        };

        (severity, confidence)
    }

    /// Predict triplet interaction.
    fn predict_triplet(
        drug_a: &Drug,
        drug_b: &Drug,
        drug_c: &Drug,
        _kb: &InteractionKnowledgeBase,
    ) -> (Severity, f64) {
        // Aggregate pairwise features
        let sim_ab = Self::cosine_similarity(&drug_a.moa_features, &drug_b.moa_features);
        let sim_bc = Self::cosine_similarity(&drug_b.moa_features, &drug_c.moa_features);
        let sim_ac = Self::cosine_similarity(&drug_a.moa_features, &drug_c.moa_features);

        // Complex interaction if all pairs have some relationship
        let avg_sim = (sim_ab + sim_bc + sim_ac) / 3.0;

        // Target overlap analysis
        let all_targets: HashSet<_> = drug_a
            .targets
            .iter()
            .chain(drug_b.targets.iter())
            .chain(drug_c.targets.iter())
            .collect();

        let unique_targets = all_targets.len();
        let total_targets = drug_a.targets.len() + drug_b.targets.len() + drug_c.targets.len();

        let overlap_ratio = if total_targets > 0 {
            1.0 - (unique_targets as f64 / total_targets as f64)
        } else {
            0.0
        };

        let risk_score = avg_sim * 0.4 + overlap_ratio * 0.6;
        let confidence = risk_score * 0.7; // Lower confidence for triplets

        let severity = if risk_score > 0.6 {
            Severity::Major
        } else if risk_score > 0.4 {
            Severity::Moderate
        } else {
            Severity::Minor
        };

        (severity, confidence)
    }

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() || b.is_empty() || a.len() != b.len() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }

        let set_a: HashSet<_> = a.iter().collect();
        let set_b: HashSet<_> = b.iter().collect();

        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn severity_to_risk(severity: Severity) -> f64 {
        match severity {
            Severity::Major => 1.0,
            Severity::Moderate => 0.6,
            Severity::Minor => 0.3,
            Severity::Minimal => 0.1,
        }
    }

    fn predict_mechanism(drug_a: &Drug, drug_b: &Drug) -> Option<String> {
        let target_overlap = Self::jaccard_similarity(&drug_a.targets, &drug_b.targets);

        if target_overlap > 0.5 {
            Some("Pharmacodynamic: competing for same targets".to_string())
        } else if target_overlap > 0.2 {
            Some("Pharmacodynamic: overlapping target pathways".to_string())
        } else {
            Some("Pharmacokinetic: potential metabolic interaction".to_string())
        }
    }

    fn calculate_polypharmacy_risk(
        drug_count: usize,
        interactions: &[PredictedInteraction],
    ) -> f64 {
        // Base risk from drug count
        let count_risk = (drug_count as f64 - 1.0).max(0.0) * 0.1;

        // Interaction-based risk
        let interaction_risk: f64 = interactions
            .iter()
            .map(|i| i.risk_score * i.confidence)
            .sum::<f64>()
            / interactions.len().max(1) as f64;

        (count_risk + interaction_risk).min(1.0)
    }

    fn generate_recommendations(
        interactions: &[PredictedInteraction],
        high_risk: &[Vec<String>],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !high_risk.is_empty() {
            recommendations.push(format!(
                "ALERT: {} high-risk drug combinations detected. Consider alternatives.",
                high_risk.len()
            ));
        }

        let major_count = interactions
            .iter()
            .filter(|i| i.severity == Severity::Major)
            .count();
        if major_count > 0 {
            recommendations.push(format!(
                "Review {} major interactions before prescribing.",
                major_count
            ));
        }

        if interactions.len() > 5 {
            recommendations
                .push("Consider medication review to reduce polypharmacy risk.".to_string());
        }

        recommendations
    }
}

impl GpuKernel for DrugInteractionPrediction {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Clinical Pathway Conformance Kernel
// ============================================================================

/// Configuration for clinical pathway conformance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayConformanceConfig {
    /// Strictness level for conformance checking.
    pub strictness: ConformanceStrictness,
    /// Allow deviations with documented reasons.
    pub allow_documented_deviations: bool,
    /// Time tolerance for step ordering (in hours).
    pub time_tolerance_hours: f64,
    /// Check required steps only or all steps.
    pub required_only: bool,
}

impl Default for PathwayConformanceConfig {
    fn default() -> Self {
        Self {
            strictness: ConformanceStrictness::Standard,
            allow_documented_deviations: true,
            time_tolerance_hours: 24.0,
            required_only: false,
        }
    }
}

/// Strictness level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConformanceStrictness {
    /// Relaxed checking, warnings only.
    Relaxed,
    /// Standard checking.
    Standard,
    /// Strict checking, all deviations flagged.
    Strict,
}

/// A step in a clinical pathway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayStep {
    /// Step identifier.
    pub id: String,
    /// Step name/description.
    pub name: String,
    /// Required step (must be completed).
    pub required: bool,
    /// Expected timing (hours from start).
    pub expected_timing: Option<f64>,
    /// Dependencies (steps that must come before).
    pub dependencies: Vec<String>,
    /// Step category.
    pub category: StepCategory,
}

/// Category of pathway step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StepCategory {
    /// Diagnostic test or assessment.
    Diagnostic,
    /// Treatment or intervention.
    Treatment,
    /// Medication administration.
    Medication,
    /// Monitoring or observation.
    Monitoring,
    /// Consultation or referral.
    Consultation,
    /// Documentation or administrative.
    Administrative,
}

/// A clinical pathway/protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalPathway {
    /// Pathway identifier.
    pub id: String,
    /// Pathway name.
    pub name: String,
    /// Condition/diagnosis this pathway applies to.
    pub condition: String,
    /// Steps in the pathway.
    pub steps: Vec<PathwayStep>,
    /// Expected total duration (hours).
    pub expected_duration_hours: f64,
}

/// A completed care event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CareEvent {
    /// Event identifier.
    pub id: String,
    /// Corresponding pathway step ID (if matched).
    pub step_id: Option<String>,
    /// Event description.
    pub description: String,
    /// Timestamp (hours from pathway start).
    pub timestamp_hours: f64,
    /// Category of the event.
    pub category: StepCategory,
    /// Deviation reason if applicable.
    pub deviation_reason: Option<String>,
}

/// A conformance deviation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayDeviation {
    /// Step that deviated.
    pub step_id: String,
    /// Type of deviation.
    pub deviation_type: DeviationType,
    /// Severity of deviation.
    pub severity: DeviationSeverity,
    /// Description.
    pub description: String,
    /// Was a reason documented.
    pub reason_documented: bool,
}

/// Type of pathway deviation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviationType {
    /// Required step was missed.
    MissedStep,
    /// Step completed out of order.
    OrderViolation,
    /// Step timing deviated significantly.
    TimingDeviation,
    /// Extra step not in pathway.
    ExtraStep,
    /// Step completed multiple times.
    DuplicateStep,
}

/// Severity of deviation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviationSeverity {
    /// Critical - safety concern.
    Critical,
    /// Major - significant protocol violation.
    Major,
    /// Minor - documented acceptable deviation.
    Minor,
    /// Informational only.
    Info,
}

/// Result of conformance check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceResult {
    /// Overall conformance score (0-1).
    pub conformance_score: f64,
    /// Is the pathway fully conformant.
    pub is_conformant: bool,
    /// List of deviations.
    pub deviations: Vec<PathwayDeviation>,
    /// Matched steps.
    pub matched_steps: Vec<String>,
    /// Unmatched pathway steps.
    pub missing_steps: Vec<String>,
    /// Events not matching any step.
    pub extra_events: Vec<String>,
    /// Completion percentage.
    pub completion_percentage: f64,
}

/// Clinical Pathway Conformance kernel.
///
/// Checks patient care events against clinical pathways/protocols
/// to identify deviations, ensure guideline adherence, and
/// support quality metrics.
#[derive(Debug, Clone)]
pub struct ClinicalPathwayConformance {
    metadata: KernelMetadata,
}

impl Default for ClinicalPathwayConformance {
    fn default() -> Self {
        Self::new()
    }
}

impl ClinicalPathwayConformance {
    /// Create a new Clinical Pathway Conformance kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch(
                "ml/clinical-pathway-conformance",
                Domain::StatisticalML,
            )
            .with_description("Clinical guideline and pathway conformance checking")
            .with_throughput(10_000)
            .with_latency_us(50.0),
        }
    }

    /// Check conformance of care events against a pathway.
    pub fn check_conformance(
        pathway: &ClinicalPathway,
        events: &[CareEvent],
        config: &PathwayConformanceConfig,
    ) -> ConformanceResult {
        if pathway.steps.is_empty() {
            return ConformanceResult {
                conformance_score: 1.0,
                is_conformant: true,
                deviations: Vec::new(),
                matched_steps: Vec::new(),
                missing_steps: Vec::new(),
                extra_events: Vec::new(),
                completion_percentage: 100.0,
            };
        }

        let mut deviations = Vec::new();
        let mut matched_steps = Vec::new();
        let mut matched_event_ids: HashSet<String> = HashSet::new();

        // Match events to steps
        for step in &pathway.steps {
            if config.required_only && !step.required {
                continue;
            }

            let matching_events: Vec<&CareEvent> = events
                .iter()
                .filter(|e| {
                    e.step_id.as_ref() == Some(&step.id)
                        || (e.category == step.category
                            && e.description
                                .to_lowercase()
                                .contains(&step.name.to_lowercase()))
                })
                .collect();

            if matching_events.is_empty() {
                if step.required {
                    deviations.push(PathwayDeviation {
                        step_id: step.id.clone(),
                        deviation_type: DeviationType::MissedStep,
                        severity: DeviationSeverity::Major,
                        description: format!("Required step '{}' was not completed", step.name),
                        reason_documented: false,
                    });
                }
            } else {
                matched_steps.push(step.id.clone());
                matched_event_ids.insert(matching_events[0].id.clone());

                // Check timing
                if let Some(expected_time) = step.expected_timing {
                    let actual_time = matching_events[0].timestamp_hours;
                    let time_diff = (actual_time - expected_time).abs();

                    if time_diff > config.time_tolerance_hours {
                        let severity = if time_diff > config.time_tolerance_hours * 2.0 {
                            DeviationSeverity::Major
                        } else {
                            DeviationSeverity::Minor
                        };

                        deviations.push(PathwayDeviation {
                            step_id: step.id.clone(),
                            deviation_type: DeviationType::TimingDeviation,
                            severity,
                            description: format!(
                                "Step '{}' timing deviation: expected {}h, actual {}h",
                                step.name, expected_time, actual_time
                            ),
                            reason_documented: matching_events[0].deviation_reason.is_some(),
                        });
                    }
                }

                // Check for duplicates
                if matching_events.len() > 1 {
                    deviations.push(PathwayDeviation {
                        step_id: step.id.clone(),
                        deviation_type: DeviationType::DuplicateStep,
                        severity: DeviationSeverity::Info,
                        description: format!(
                            "Step '{}' completed {} times",
                            step.name,
                            matching_events.len()
                        ),
                        reason_documented: true,
                    });
                }
            }
        }

        // Check dependencies (ordering)
        for step in &pathway.steps {
            if !matched_steps.contains(&step.id) {
                continue;
            }

            for dep_id in &step.dependencies {
                if !matched_steps.contains(dep_id) {
                    deviations.push(PathwayDeviation {
                        step_id: step.id.clone(),
                        deviation_type: DeviationType::OrderViolation,
                        severity: DeviationSeverity::Major,
                        description: format!(
                            "Step '{}' completed before dependency '{}'",
                            step.name, dep_id
                        ),
                        reason_documented: false,
                    });
                }
            }
        }

        // Find extra events
        let extra_events: Vec<String> = events
            .iter()
            .filter(|e| !matched_event_ids.contains(&e.id))
            .map(|e| e.id.clone())
            .collect();

        // Calculate missing steps
        let required_steps: Vec<_> = pathway
            .steps
            .iter()
            .filter(|s| s.required)
            .map(|s| s.id.clone())
            .collect();

        let missing_steps: Vec<String> = required_steps
            .iter()
            .filter(|s| !matched_steps.contains(s))
            .cloned()
            .collect();

        // Apply documented deviation allowance
        if config.allow_documented_deviations {
            deviations
                .retain(|d| !(d.reason_documented && d.severity != DeviationSeverity::Critical));
        }

        // Calculate scores
        let completion_percentage = if required_steps.is_empty() {
            100.0
        } else {
            (matched_steps.len() as f64 / required_steps.len() as f64) * 100.0
        };

        let deviation_penalty: f64 = deviations
            .iter()
            .map(|d| match d.severity {
                DeviationSeverity::Critical => 0.4,
                DeviationSeverity::Major => 0.2,
                DeviationSeverity::Minor => 0.05,
                DeviationSeverity::Info => 0.0,
            })
            .sum();

        let conformance_score =
            (1.0 - deviation_penalty).max(0.0) * (completion_percentage / 100.0);

        let is_conformant = match config.strictness {
            ConformanceStrictness::Relaxed => conformance_score >= 0.7,
            ConformanceStrictness::Standard => {
                conformance_score >= 0.85 && missing_steps.is_empty()
            }
            ConformanceStrictness::Strict => {
                conformance_score >= 0.95
                    && missing_steps.is_empty()
                    && deviations
                        .iter()
                        .all(|d| d.severity == DeviationSeverity::Info)
            }
        };

        ConformanceResult {
            conformance_score,
            is_conformant,
            deviations,
            matched_steps,
            missing_steps,
            extra_events,
            completion_percentage,
        }
    }

    /// Check multiple patients against the same pathway.
    pub fn check_batch(
        pathway: &ClinicalPathway,
        patient_events: &[Vec<CareEvent>],
        config: &PathwayConformanceConfig,
    ) -> Vec<ConformanceResult> {
        patient_events
            .iter()
            .map(|events| Self::check_conformance(pathway, events, config))
            .collect()
    }

    /// Calculate aggregate statistics across patients.
    pub fn aggregate_stats(results: &[ConformanceResult]) -> PathwayStatistics {
        if results.is_empty() {
            return PathwayStatistics::default();
        }

        let n = results.len() as f64;

        let avg_conformance = results.iter().map(|r| r.conformance_score).sum::<f64>() / n;
        let conformant_count = results.iter().filter(|r| r.is_conformant).count();
        let avg_completion = results.iter().map(|r| r.completion_percentage).sum::<f64>() / n;

        // Count deviation types
        let mut deviation_counts: HashMap<DeviationType, usize> = HashMap::new();
        for result in results {
            for dev in &result.deviations {
                *deviation_counts.entry(dev.deviation_type).or_insert(0) += 1;
            }
        }

        PathwayStatistics {
            total_patients: results.len(),
            conformant_patients: conformant_count,
            conformance_rate: conformant_count as f64 / n,
            average_conformance_score: avg_conformance,
            average_completion: avg_completion,
            deviation_counts,
        }
    }
}

/// Aggregate pathway statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathwayStatistics {
    /// Total patients analyzed.
    pub total_patients: usize,
    /// Patients meeting conformance threshold.
    pub conformant_patients: usize,
    /// Conformance rate (0-1).
    pub conformance_rate: f64,
    /// Average conformance score.
    pub average_conformance_score: f64,
    /// Average completion percentage.
    pub average_completion: f64,
    /// Count of each deviation type.
    pub deviation_counts: HashMap<DeviationType, usize>,
}

impl GpuKernel for ClinicalPathwayConformance {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drug_interaction_metadata() {
        let kernel = DrugInteractionPrediction::new();
        assert_eq!(kernel.metadata().id, "ml/drug-interaction-prediction");
    }

    #[test]
    fn test_drug_interaction_basic() {
        let drugs = vec![
            Drug {
                id: "drug1".to_string(),
                name: "Aspirin".to_string(),
                drug_class: Some("NSAID".to_string()),
                moa_features: vec![1.0, 0.0, 0.0],
                targets: vec!["COX1".to_string(), "COX2".to_string()],
            },
            Drug {
                id: "drug2".to_string(),
                name: "Ibuprofen".to_string(),
                drug_class: Some("NSAID".to_string()),
                // Different MOA (orthogonal) + same targets = high risk
                moa_features: vec![0.0, 1.0, 0.0],
                targets: vec!["COX1".to_string(), "COX2".to_string()],
            },
        ];

        let kb = InteractionKnowledgeBase::default();
        let config = DrugInteractionConfig::default();

        let result = DrugInteractionPrediction::predict(&drugs, &kb, &config);

        // Should detect potential interaction (same targets + different MOA)
        assert!(!result.interactions.is_empty());
    }

    #[test]
    fn test_known_interaction() {
        let drugs = vec![
            Drug {
                id: "warfarin".to_string(),
                name: "Warfarin".to_string(),
                drug_class: Some("Anticoagulant".to_string()),
                moa_features: vec![],
                targets: vec![],
            },
            Drug {
                id: "aspirin".to_string(),
                name: "Aspirin".to_string(),
                drug_class: Some("NSAID".to_string()),
                moa_features: vec![],
                targets: vec![],
            },
        ];

        let mut kb = InteractionKnowledgeBase::default();
        kb.add_interaction(KnownInteraction {
            drug_ids: vec!["warfarin".to_string(), "aspirin".to_string()],
            severity: Severity::Major,
            description: "Increased bleeding risk".to_string(),
            recommendation: "Avoid combination".to_string(),
        });

        let config = DrugInteractionConfig::default();
        let result = DrugInteractionPrediction::predict(&drugs, &kb, &config);

        assert!(
            result
                .interactions
                .iter()
                .any(|i| i.is_known && i.severity == Severity::Major)
        );
    }

    #[test]
    fn test_empty_drugs() {
        let kb = InteractionKnowledgeBase::default();
        let config = DrugInteractionConfig::default();
        let result = DrugInteractionPrediction::predict(&[], &kb, &config);
        assert!(result.interactions.is_empty());
    }

    #[test]
    fn test_pathway_conformance_metadata() {
        let kernel = ClinicalPathwayConformance::new();
        assert_eq!(kernel.metadata().id, "ml/clinical-pathway-conformance");
    }

    #[test]
    fn test_pathway_conformance_basic() {
        let pathway = ClinicalPathway {
            id: "sepsis".to_string(),
            name: "Sepsis Bundle".to_string(),
            condition: "Sepsis".to_string(),
            steps: vec![
                PathwayStep {
                    id: "lactate".to_string(),
                    name: "Measure lactate".to_string(),
                    required: true,
                    expected_timing: Some(1.0),
                    dependencies: vec![],
                    category: StepCategory::Diagnostic,
                },
                PathwayStep {
                    id: "cultures".to_string(),
                    name: "Blood cultures".to_string(),
                    required: true,
                    expected_timing: Some(1.0),
                    dependencies: vec![],
                    category: StepCategory::Diagnostic,
                },
                PathwayStep {
                    id: "antibiotics".to_string(),
                    name: "Broad spectrum antibiotics".to_string(),
                    required: true,
                    expected_timing: Some(3.0),
                    dependencies: vec!["cultures".to_string()],
                    category: StepCategory::Medication,
                },
            ],
            expected_duration_hours: 6.0,
        };

        let events = vec![
            CareEvent {
                id: "e1".to_string(),
                step_id: Some("lactate".to_string()),
                description: "Lactate measured".to_string(),
                timestamp_hours: 0.5,
                category: StepCategory::Diagnostic,
                deviation_reason: None,
            },
            CareEvent {
                id: "e2".to_string(),
                step_id: Some("cultures".to_string()),
                description: "Blood cultures drawn".to_string(),
                timestamp_hours: 0.75,
                category: StepCategory::Diagnostic,
                deviation_reason: None,
            },
            CareEvent {
                id: "e3".to_string(),
                step_id: Some("antibiotics".to_string()),
                description: "Antibiotics administered".to_string(),
                timestamp_hours: 2.0,
                category: StepCategory::Medication,
                deviation_reason: None,
            },
        ];

        let config = PathwayConformanceConfig::default();
        let result = ClinicalPathwayConformance::check_conformance(&pathway, &events, &config);

        assert!(result.conformance_score > 0.9);
        assert!(result.is_conformant);
        assert!(result.missing_steps.is_empty());
    }

    #[test]
    fn test_missed_required_step() {
        let pathway = ClinicalPathway {
            id: "test".to_string(),
            name: "Test".to_string(),
            condition: "Test".to_string(),
            steps: vec![PathwayStep {
                id: "required".to_string(),
                name: "Required step".to_string(),
                required: true,
                expected_timing: None,
                dependencies: vec![],
                category: StepCategory::Treatment,
            }],
            expected_duration_hours: 24.0,
        };

        let events: Vec<CareEvent> = vec![];
        let config = PathwayConformanceConfig::default();

        let result = ClinicalPathwayConformance::check_conformance(&pathway, &events, &config);

        assert!(!result.is_conformant);
        assert!(
            result
                .deviations
                .iter()
                .any(|d| d.deviation_type == DeviationType::MissedStep)
        );
    }

    #[test]
    fn test_batch_conformance() {
        let pathway = ClinicalPathway {
            id: "simple".to_string(),
            name: "Simple".to_string(),
            condition: "Test".to_string(),
            steps: vec![PathwayStep {
                id: "step1".to_string(),
                name: "Step 1".to_string(),
                required: true,
                expected_timing: None,
                dependencies: vec![],
                category: StepCategory::Treatment,
            }],
            expected_duration_hours: 24.0,
        };

        let patient_events = vec![
            vec![CareEvent {
                id: "p1e1".to_string(),
                step_id: Some("step1".to_string()),
                description: "Step 1".to_string(),
                timestamp_hours: 1.0,
                category: StepCategory::Treatment,
                deviation_reason: None,
            }],
            vec![], // Non-conformant
        ];

        let config = PathwayConformanceConfig::default();
        let results = ClinicalPathwayConformance::check_batch(&pathway, &patient_events, &config);

        assert_eq!(results.len(), 2);
        assert!(results[0].is_conformant);
        assert!(!results[1].is_conformant);

        let stats = ClinicalPathwayConformance::aggregate_stats(&results);
        assert_eq!(stats.total_patients, 2);
        assert_eq!(stats.conformant_patients, 1);
    }
}
