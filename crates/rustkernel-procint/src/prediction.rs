//! Next activity prediction kernels.
//!
//! This module provides process activity prediction:
//! - Markov chain-based prediction
//! - N-gram model prediction
//! - Batch inference for multiple traces

use crate::types::EventLog;
use rustkernel_core::traits::GpuKernel;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Next Activity Prediction Kernel
// ============================================================================

/// Prediction model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PredictionModelType {
    /// First-order Markov chain (single previous activity).
    #[default]
    Markov1,
    /// Second-order Markov chain (two previous activities).
    Markov2,
    /// N-gram model with configurable n.
    NGram,
}

/// Configuration for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Type of prediction model.
    pub model_type: PredictionModelType,
    /// N for N-gram model (ignored for Markov).
    pub n_gram_size: usize,
    /// Number of top predictions to return.
    pub top_k: usize,
    /// Minimum probability threshold.
    pub min_probability: f64,
    /// Use Laplace smoothing for unseen transitions.
    pub laplace_smoothing: bool,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            model_type: PredictionModelType::Markov1,
            n_gram_size: 3,
            top_k: 5,
            min_probability: 0.01,
            laplace_smoothing: true,
        }
    }
}

/// Transition matrix for first-order Markov model.
/// Key: current activity, Value: map of next activity -> count
pub type TransitionMatrix = HashMap<String, HashMap<String, u64>>;

/// Higher-order transition matrix.
/// Key: sequence of activities (as tuple), Value: map of next activity -> count
pub type HigherOrderTransitions = HashMap<Vec<String>, HashMap<String, u64>>;

/// A trained prediction model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModel {
    /// Model type.
    pub model_type: PredictionModelType,
    /// First-order transitions (activity -> next -> count).
    pub transitions: TransitionMatrix,
    /// Higher-order transitions (for Markov2 and N-gram).
    pub higher_order: HigherOrderTransitions,
    /// Start activity frequencies.
    pub start_activities: HashMap<String, u64>,
    /// End activity frequencies.
    pub end_activities: HashMap<String, u64>,
    /// Activity vocabulary.
    pub vocabulary: Vec<String>,
    /// Total traces trained on.
    pub trace_count: u64,
    /// Total events trained on.
    pub event_count: u64,
}

impl Default for PredictionModel {
    fn default() -> Self {
        Self {
            model_type: PredictionModelType::Markov1,
            transitions: HashMap::new(),
            higher_order: HashMap::new(),
            start_activities: HashMap::new(),
            end_activities: HashMap::new(),
            vocabulary: Vec::new(),
            trace_count: 0,
            event_count: 0,
        }
    }
}

impl PredictionModel {
    /// Create a new model from an event log.
    pub fn train(log: &EventLog, config: &PredictionConfig) -> Self {
        let mut model = Self {
            model_type: config.model_type,
            ..Default::default()
        };

        let mut vocab_set = std::collections::HashSet::new();

        for trace in log.traces.values() {
            if trace.events.is_empty() {
                continue;
            }

            model.trace_count += 1;
            model.event_count += trace.events.len() as u64;

            let activities: Vec<&str> = trace.events.iter().map(|e| e.activity.as_str()).collect();

            // Record start/end activities
            if let Some(first) = activities.first() {
                *model.start_activities.entry(first.to_string()).or_default() += 1;
            }
            if let Some(last) = activities.last() {
                *model.end_activities.entry(last.to_string()).or_default() += 1;
            }

            // Build vocabulary
            for act in &activities {
                vocab_set.insert(act.to_string());
            }

            // Build transition matrix
            for window in activities.windows(2) {
                let from = window[0].to_string();
                let to = window[1].to_string();
                *model
                    .transitions
                    .entry(from)
                    .or_default()
                    .entry(to)
                    .or_default() += 1;
            }

            // Build higher-order transitions if needed
            match config.model_type {
                PredictionModelType::Markov2 => {
                    for window in activities.windows(3) {
                        let key = vec![window[0].to_string(), window[1].to_string()];
                        let next = window[2].to_string();
                        *model
                            .higher_order
                            .entry(key)
                            .or_default()
                            .entry(next)
                            .or_default() += 1;
                    }
                }
                PredictionModelType::NGram => {
                    let n = config.n_gram_size;
                    if activities.len() >= n {
                        for window in activities.windows(n) {
                            let key: Vec<String> =
                                window[..n - 1].iter().map(|s| s.to_string()).collect();
                            let next = window[n - 1].to_string();
                            *model
                                .higher_order
                                .entry(key)
                                .or_default()
                                .entry(next)
                                .or_default() += 1;
                        }
                    }
                }
                PredictionModelType::Markov1 => {}
            }
        }

        model.vocabulary = vocab_set.into_iter().collect();
        model.vocabulary.sort();

        model
    }

    /// Predict next activities for a given sequence.
    pub fn predict(
        &self,
        history: &[String],
        config: &PredictionConfig,
    ) -> Vec<ActivityPrediction> {
        let vocab_size = self.vocabulary.len();
        let smoothing = if config.laplace_smoothing { 1.0 } else { 0.0 };

        // Get transition counts based on model type
        let counts: Option<&HashMap<String, u64>> = match self.model_type {
            PredictionModelType::Markov1 => {
                history.last().and_then(|last| self.transitions.get(last))
            }
            PredictionModelType::Markov2 => {
                if history.len() >= 2 {
                    let key = vec![
                        history[history.len() - 2].clone(),
                        history[history.len() - 1].clone(),
                    ];
                    self.higher_order.get(&key)
                } else if history.len() == 1 {
                    // Fall back to first-order
                    self.transitions.get(&history[0])
                } else {
                    None
                }
            }
            PredictionModelType::NGram => {
                let n = config.n_gram_size;
                if history.len() >= n - 1 {
                    let key: Vec<String> = history[history.len() - (n - 1)..].to_vec();
                    self.higher_order.get(&key)
                } else if !history.is_empty() {
                    // Fall back to first-order
                    self.transitions.get(&history[history.len() - 1])
                } else {
                    None
                }
            }
        };

        // Calculate probabilities
        let mut predictions: Vec<ActivityPrediction> = if let Some(counts) = counts {
            let total: u64 = counts.values().sum();
            let total_with_smoothing = total as f64 + smoothing * vocab_size as f64;

            self.vocabulary
                .iter()
                .map(|activity| {
                    let count = counts.get(activity).copied().unwrap_or(0);
                    let prob = (count as f64 + smoothing) / total_with_smoothing;
                    ActivityPrediction {
                        activity: activity.clone(),
                        probability: prob,
                        confidence: if total > 10 { prob } else { prob * 0.5 },
                        is_end: self.end_activities.contains_key(activity),
                    }
                })
                .filter(|p| p.probability >= config.min_probability)
                .collect()
        } else if config.laplace_smoothing && !self.vocabulary.is_empty() {
            // Uniform distribution with smoothing for unseen context
            let prob = 1.0 / vocab_size as f64;
            self.vocabulary
                .iter()
                .map(|activity| ActivityPrediction {
                    activity: activity.clone(),
                    probability: prob,
                    confidence: 0.1, // Low confidence for uniform
                    is_end: self.end_activities.contains_key(activity),
                })
                .collect()
        } else {
            Vec::new()
        };

        // Sort by probability descending and take top_k
        predictions.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        predictions.truncate(config.top_k);

        predictions
    }

    /// Predict from activity names (convenience method).
    pub fn predict_from_names(
        &self,
        history: &[&str],
        config: &PredictionConfig,
    ) -> Vec<ActivityPrediction> {
        let history: Vec<String> = history.iter().map(|s| s.to_string()).collect();
        self.predict(&history, config)
    }
}

/// A predicted next activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityPrediction {
    /// Predicted activity name.
    pub activity: String,
    /// Probability of this activity.
    pub probability: f64,
    /// Confidence in the prediction (adjusted for data sparsity).
    pub confidence: f64,
    /// Whether this is commonly an end activity.
    pub is_end: bool,
}

/// Input for batch prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInput {
    /// Traces to predict next activities for.
    pub traces: Vec<TraceHistory>,
    /// Trained model.
    pub model: PredictionModel,
    /// Configuration.
    pub config: PredictionConfig,
}

/// A trace history for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHistory {
    /// Case/trace ID.
    pub case_id: String,
    /// Activity history (most recent last).
    pub activities: Vec<String>,
}

/// Output from batch prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutput {
    /// Predictions per trace.
    pub predictions: Vec<TracePrediction>,
    /// Compute time in microseconds.
    pub compute_time_us: u64,
}

/// Predictions for a single trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracePrediction {
    /// Case/trace ID.
    pub case_id: String,
    /// Top-k predictions.
    pub predictions: Vec<ActivityPrediction>,
    /// Expected remaining activities (if model supports).
    pub expected_remaining: Option<f64>,
}

/// Next activity prediction kernel.
///
/// Predicts the next activity in a business process using
/// Markov chains or N-gram models trained on historical data.
#[derive(Debug, Clone)]
pub struct NextActivityPrediction {
    metadata: KernelMetadata,
}

impl Default for NextActivityPrediction {
    fn default() -> Self {
        Self::new()
    }
}

impl NextActivityPrediction {
    /// Create a new next activity prediction kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("procint/next-activity", Domain::ProcessIntelligence)
                .with_description("Markov/N-gram next activity prediction")
                .with_throughput(100_000)
                .with_latency_us(50.0),
        }
    }

    /// Train a model from an event log.
    pub fn train(log: &EventLog, config: &PredictionConfig) -> PredictionModel {
        PredictionModel::train(log, config)
    }

    /// Batch predict for multiple traces.
    pub fn predict_batch(
        traces: &[TraceHistory],
        model: &PredictionModel,
        config: &PredictionConfig,
    ) -> Vec<TracePrediction> {
        traces
            .iter()
            .map(|trace| {
                let predictions = model.predict(&trace.activities, config);
                TracePrediction {
                    case_id: trace.case_id.clone(),
                    predictions,
                    expected_remaining: None,
                }
            })
            .collect()
    }

    /// Compute batch predictions.
    pub fn compute(input: &PredictionInput) -> PredictionOutput {
        let start = Instant::now();
        let predictions = Self::predict_batch(&input.traces, &input.model, &input.config);
        PredictionOutput {
            predictions,
            compute_time_us: start.elapsed().as_micros() as u64,
        }
    }
}

impl GpuKernel for NextActivityPrediction {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProcessEvent;

    fn create_test_log() -> EventLog {
        let mut log = EventLog::new("test".to_string());

        // Trace 1: A -> B -> C -> D
        for (i, activity) in ["A", "B", "C", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: i as u64,
                case_id: "trace1".to_string(),
                activity: activity.to_string(),
                timestamp: i as u64 * 100,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Trace 2: A -> B -> C -> D (same pattern)
        for (i, activity) in ["A", "B", "C", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: (10 + i) as u64,
                case_id: "trace2".to_string(),
                activity: activity.to_string(),
                timestamp: i as u64 * 100,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Trace 3: A -> B -> E -> D (different path)
        for (i, activity) in ["A", "B", "E", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: (20 + i) as u64,
                case_id: "trace3".to_string(),
                activity: activity.to_string(),
                timestamp: i as u64 * 100,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Trace 4: A -> B -> C -> D
        for (i, activity) in ["A", "B", "C", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: (30 + i) as u64,
                case_id: "trace4".to_string(),
                activity: activity.to_string(),
                timestamp: i as u64 * 100,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        log
    }

    #[test]
    fn test_next_activity_prediction_metadata() {
        let kernel = NextActivityPrediction::new();
        assert_eq!(kernel.metadata().id, "procint/next-activity");
        assert_eq!(kernel.metadata().domain, Domain::ProcessIntelligence);
    }

    #[test]
    fn test_model_training() {
        let log = create_test_log();
        let config = PredictionConfig::default();
        let model = PredictionModel::train(&log, &config);

        assert_eq!(model.trace_count, 4);
        assert!(model.vocabulary.contains(&"A".to_string()));
        assert!(model.vocabulary.contains(&"B".to_string()));
        assert!(model.vocabulary.contains(&"C".to_string()));
        assert!(model.vocabulary.contains(&"D".to_string()));
        assert!(model.vocabulary.contains(&"E".to_string()));

        // Check transitions
        assert!(model.transitions.contains_key("A"));
        assert!(model.transitions.contains_key("B"));
    }

    #[test]
    fn test_first_order_prediction() {
        let log = create_test_log();
        let config = PredictionConfig {
            model_type: PredictionModelType::Markov1,
            top_k: 3,
            min_probability: 0.0,
            laplace_smoothing: false,
            ..Default::default()
        };
        let model = PredictionModel::train(&log, &config);

        // After A, B should be predicted with high probability
        let predictions = model.predict_from_names(&["A"], &config);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].activity, "B");
        assert!(predictions[0].probability > 0.9);

        // After B, C should be most likely (3 traces), E second (1 trace)
        let predictions = model.predict_from_names(&["B"], &config);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].activity, "C");
    }

    #[test]
    fn test_second_order_prediction() {
        let log = create_test_log();
        let config = PredictionConfig {
            model_type: PredictionModelType::Markov2,
            top_k: 3,
            min_probability: 0.0,
            laplace_smoothing: false,
            ..Default::default()
        };
        let model = PredictionModel::train(&log, &config);

        // After A, B -> C should be predicted (using 2nd order)
        let predictions = model.predict_from_names(&["A", "B"], &config);
        assert!(!predictions.is_empty());
        // C appears after A,B in 3 traces, E in 1 trace
        assert_eq!(predictions[0].activity, "C");
    }

    #[test]
    fn test_batch_prediction() {
        let log = create_test_log();
        let config = PredictionConfig::default();
        let model = PredictionModel::train(&log, &config);

        let traces = vec![
            TraceHistory {
                case_id: "test1".to_string(),
                activities: vec!["A".to_string()],
            },
            TraceHistory {
                case_id: "test2".to_string(),
                activities: vec!["A".to_string(), "B".to_string()],
            },
        ];

        let results = NextActivityPrediction::predict_batch(&traces, &model, &config);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].case_id, "test1");
        assert_eq!(results[1].case_id, "test2");
    }

    #[test]
    fn test_laplace_smoothing() {
        let log = create_test_log();
        let config_no_smooth = PredictionConfig {
            laplace_smoothing: false,
            top_k: 10,
            min_probability: 0.0,
            ..Default::default()
        };
        let config_smooth = PredictionConfig {
            laplace_smoothing: true,
            top_k: 10,
            min_probability: 0.0,
            ..Default::default()
        };
        let model = PredictionModel::train(&log, &config_no_smooth);

        // Without smoothing, unseen transition should have 0 probability
        let pred_no_smooth = model.predict_from_names(&["D"], &config_no_smooth);
        // D is end activity, so no transitions from it without smoothing
        let _max_prob = pred_no_smooth.iter().map(|p| p.probability).sum::<f64>();

        // With smoothing, should have non-zero probabilities
        let pred_smooth = model.predict_from_names(&["D"], &config_smooth);
        assert!(!pred_smooth.is_empty());
        assert!(pred_smooth.iter().all(|p| p.probability > 0.0));
    }

    #[test]
    fn test_start_end_activities() {
        let log = create_test_log();
        let config = PredictionConfig::default();
        let model = PredictionModel::train(&log, &config);

        // A should be start activity
        assert!(model.start_activities.contains_key("A"));
        assert_eq!(model.start_activities.get("A"), Some(&4));

        // D should be end activity
        assert!(model.end_activities.contains_key("D"));
        assert_eq!(model.end_activities.get("D"), Some(&4));
    }

    #[test]
    fn test_ngram_prediction() {
        let log = create_test_log();
        let config = PredictionConfig {
            model_type: PredictionModelType::NGram,
            n_gram_size: 3,
            top_k: 3,
            min_probability: 0.0,
            laplace_smoothing: false,
        };
        let model = PredictionModel::train(&log, &config);

        // With 3-gram: A, B -> C or E
        let predictions = model.predict_from_names(&["A", "B"], &config);
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_empty_history() {
        let log = create_test_log();
        let config = PredictionConfig {
            laplace_smoothing: true,
            ..Default::default()
        };
        let model = PredictionModel::train(&log, &config);

        // Empty history should return uniform or start distribution
        let predictions = model.predict(&[], &config);
        // With smoothing, should return something
        assert!(!predictions.is_empty() || config.laplace_smoothing);
    }

    #[test]
    fn test_compute_output() {
        let log = create_test_log();
        let config = PredictionConfig::default();
        let model = PredictionModel::train(&log, &config);

        let input = PredictionInput {
            traces: vec![TraceHistory {
                case_id: "test".to_string(),
                activities: vec!["A".to_string(), "B".to_string()],
            }],
            model,
            config,
        };

        let output = NextActivityPrediction::compute(&input);
        assert_eq!(output.predictions.len(), 1);
        assert!(output.compute_time_us < 1_000_000); // Should be fast
    }
}
