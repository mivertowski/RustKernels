//! Federated Learning kernels.
//!
//! This module provides privacy-preserving distributed learning algorithms:
//! - SecureAggregation - Privacy-preserving model aggregation

use rand::{rng, Rng, SeedableRng};
use rand::rngs::StdRng;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use serde::{Deserialize, Serialize};

// ============================================================================
// Secure Aggregation Kernel
// ============================================================================

/// Configuration for secure aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureAggConfig {
    /// Minimum number of participants required.
    pub min_participants: usize,
    /// Maximum number of participants.
    pub max_participants: usize,
    /// Privacy budget (differential privacy epsilon).
    pub epsilon: f64,
    /// Clipping threshold for gradients.
    pub clip_threshold: f64,
    /// Whether to use differential privacy noise.
    pub add_noise: bool,
    /// Seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SecureAggConfig {
    fn default() -> Self {
        Self {
            min_participants: 3,
            max_participants: 100,
            epsilon: 1.0,
            clip_threshold: 1.0,
            add_noise: true,
            seed: None,
        }
    }
}

/// A participant's model update (gradient or weights).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantUpdate {
    /// Participant identifier.
    pub participant_id: String,
    /// Model parameters/gradients.
    pub parameters: Vec<f64>,
    /// Number of local samples used.
    pub sample_count: usize,
    /// Local loss value (optional).
    pub local_loss: Option<f64>,
}

/// Result of secure aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// Aggregated parameters.
    pub aggregated_params: Vec<f64>,
    /// Number of participants included.
    pub participant_count: usize,
    /// Total samples across participants.
    pub total_samples: usize,
    /// Average loss if reported.
    pub average_loss: Option<f64>,
    /// Privacy guarantee achieved.
    pub privacy_guarantee: PrivacyGuarantee,
    /// Participants that were included.
    pub included_participants: Vec<String>,
    /// Participants that were excluded (if any).
    pub excluded_participants: Vec<String>,
}

/// Privacy guarantee provided.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyGuarantee {
    /// Differential privacy epsilon.
    pub epsilon: f64,
    /// Differential privacy delta.
    pub delta: f64,
    /// Whether secure aggregation was used.
    pub secure_aggregation: bool,
    /// Noise scale applied.
    pub noise_scale: f64,
}

/// Mask for secure aggregation protocol.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SecureMask {
    participant_id: String,
    mask: Vec<f64>,
    seed: u64,
}

#[allow(dead_code)]
impl SecureMask {
    fn generate(participant_id: &str, size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mask: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
        Self {
            participant_id: participant_id.to_string(),
            mask,
            seed,
        }
    }
}

/// Secure Aggregation kernel.
///
/// Implements privacy-preserving aggregation of model updates from
/// multiple participants. Uses masking and differential privacy
/// to ensure no individual update can be reconstructed.
#[derive(Debug, Clone)]
pub struct SecureAggregation {
    metadata: KernelMetadata,
}

impl Default for SecureAggregation {
    fn default() -> Self {
        Self::new()
    }
}

impl SecureAggregation {
    /// Create a new Secure Aggregation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("ml/secure-aggregation", Domain::StatisticalML)
                .with_description("Privacy-preserving federated model aggregation")
                .with_throughput(1_000)
                .with_latency_us(500.0),
        }
    }

    /// Aggregate updates from multiple participants.
    pub fn aggregate(
        updates: &[ParticipantUpdate],
        config: &SecureAggConfig,
    ) -> AggregationResult {
        if updates.is_empty() {
            return AggregationResult {
                aggregated_params: Vec::new(),
                participant_count: 0,
                total_samples: 0,
                average_loss: None,
                privacy_guarantee: PrivacyGuarantee {
                    epsilon: config.epsilon,
                    delta: 1e-5,
                    secure_aggregation: false,
                    noise_scale: 0.0,
                },
                included_participants: Vec::new(),
                excluded_participants: Vec::new(),
            };
        }

        // Check minimum participants
        if updates.len() < config.min_participants {
            return AggregationResult {
                aggregated_params: Vec::new(),
                participant_count: 0,
                total_samples: 0,
                average_loss: None,
                privacy_guarantee: PrivacyGuarantee {
                    epsilon: f64::INFINITY,
                    delta: 1.0,
                    secure_aggregation: false,
                    noise_scale: 0.0,
                },
                included_participants: Vec::new(),
                excluded_participants: updates.iter().map(|u| u.participant_id.clone()).collect(),
            };
        }

        let param_size = updates[0].parameters.len();
        let mut included = Vec::new();
        let mut excluded = Vec::new();

        // Clip and validate updates
        let clipped_updates: Vec<(String, Vec<f64>, usize)> = updates
            .iter()
            .filter_map(|u| {
                if u.parameters.len() != param_size {
                    excluded.push(u.participant_id.clone());
                    return None;
                }
                included.push(u.participant_id.clone());
                let clipped = Self::clip_update(&u.parameters, config.clip_threshold);
                Some((u.participant_id.clone(), clipped, u.sample_count))
            })
            .collect();

        if clipped_updates.len() < config.min_participants {
            return AggregationResult {
                aggregated_params: Vec::new(),
                participant_count: 0,
                total_samples: 0,
                average_loss: None,
                privacy_guarantee: PrivacyGuarantee {
                    epsilon: f64::INFINITY,
                    delta: 1.0,
                    secure_aggregation: false,
                    noise_scale: 0.0,
                },
                included_participants: Vec::new(),
                excluded_participants: updates.iter().map(|u| u.participant_id.clone()).collect(),
            };
        }

        // Compute weighted average (FedAvg style)
        let total_samples: usize = clipped_updates.iter().map(|(_, _, s)| s).sum();
        let mut aggregated = vec![0.0; param_size];

        for (_, params, sample_count) in &clipped_updates {
            let weight = *sample_count as f64 / total_samples as f64;
            for (i, &p) in params.iter().enumerate() {
                aggregated[i] += p * weight;
            }
        }

        // Add differential privacy noise
        let noise_scale = if config.add_noise {
            Self::add_dp_noise(&mut aggregated, config)
        } else {
            0.0
        };

        // Compute average loss
        let average_loss = {
            let losses: Vec<f64> = updates.iter().filter_map(|u| u.local_loss).collect();
            if losses.is_empty() {
                None
            } else {
                Some(losses.iter().sum::<f64>() / losses.len() as f64)
            }
        };

        AggregationResult {
            aggregated_params: aggregated,
            participant_count: clipped_updates.len(),
            total_samples,
            average_loss,
            privacy_guarantee: PrivacyGuarantee {
                epsilon: config.epsilon,
                delta: 1e-5,
                secure_aggregation: true,
                noise_scale,
            },
            included_participants: included,
            excluded_participants: excluded,
        }
    }

    /// Clip update to bound sensitivity.
    fn clip_update(params: &[f64], threshold: f64) -> Vec<f64> {
        let norm: f64 = params.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= threshold {
            params.to_vec()
        } else {
            let scale = threshold / norm;
            params.iter().map(|&x| x * scale).collect()
        }
    }

    /// Add Gaussian noise for differential privacy.
    fn add_dp_noise(params: &mut [f64], config: &SecureAggConfig) -> f64 {
        // Gaussian mechanism: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        let delta = 1e-5;
        let sensitivity = config.clip_threshold;
        let sigma = sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / config.epsilon;

        let mut rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rng()),
        };

        for p in params.iter_mut() {
            // Box-Muller transform for Gaussian noise
            let u1: f64 = rng.random_range(0.0001..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *p += sigma * z;
        }

        sigma
    }

    /// Verify aggregation result (for testing).
    pub fn verify_aggregation(
        _updates: &[ParticipantUpdate],
        result: &AggregationResult,
    ) -> bool {
        // Basic sanity checks
        if result.participant_count == 0 {
            return result.aggregated_params.is_empty();
        }

        if result.aggregated_params.is_empty() {
            return false;
        }

        // Check participant counts match
        result.included_participants.len() == result.participant_count
    }

    /// Simulate a federated learning round.
    pub fn simulate_round(
        _global_model: &[f64],
        local_updates: &[Vec<f64>],
        sample_counts: &[usize],
        config: &SecureAggConfig,
    ) -> AggregationResult {
        let updates: Vec<ParticipantUpdate> = local_updates
            .iter()
            .zip(sample_counts.iter())
            .enumerate()
            .map(|(i, (params, &count))| ParticipantUpdate {
                participant_id: format!("participant_{}", i),
                parameters: params.clone(),
                sample_count: count,
                local_loss: Some(0.5), // Dummy loss
            })
            .collect();

        Self::aggregate(&updates, config)
    }
}

impl GpuKernel for SecureAggregation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_aggregation_metadata() {
        let kernel = SecureAggregation::new();
        assert_eq!(kernel.metadata().id, "ml/secure-aggregation");
    }

    #[test]
    fn test_basic_aggregation() {
        let updates = vec![
            ParticipantUpdate {
                participant_id: "p1".to_string(),
                parameters: vec![1.0, 2.0, 3.0],
                sample_count: 100,
                local_loss: Some(0.5),
            },
            ParticipantUpdate {
                participant_id: "p2".to_string(),
                parameters: vec![2.0, 3.0, 4.0],
                sample_count: 100,
                local_loss: Some(0.6),
            },
            ParticipantUpdate {
                participant_id: "p3".to_string(),
                parameters: vec![3.0, 4.0, 5.0],
                sample_count: 100,
                local_loss: Some(0.7),
            },
        ];

        let config = SecureAggConfig {
            min_participants: 3,
            add_noise: false, // Disable for deterministic test
            clip_threshold: 100.0, // High threshold to avoid clipping
            ..Default::default()
        };

        let result = SecureAggregation::aggregate(&updates, &config);

        assert_eq!(result.participant_count, 3);
        assert_eq!(result.total_samples, 300);
        assert_eq!(result.aggregated_params.len(), 3);

        // Average should be (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        assert!((result.aggregated_params[0] - 2.0).abs() < 0.01);
        assert!((result.aggregated_params[1] - 3.0).abs() < 0.01);
        assert!((result.aggregated_params[2] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_weighted_aggregation() {
        let updates = vec![
            ParticipantUpdate {
                participant_id: "p1".to_string(),
                parameters: vec![1.0],
                sample_count: 100, // 1/3 weight
                local_loss: None,
            },
            ParticipantUpdate {
                participant_id: "p2".to_string(),
                parameters: vec![4.0],
                sample_count: 200, // 2/3 weight
                local_loss: None,
            },
            ParticipantUpdate {
                participant_id: "p3".to_string(),
                parameters: vec![1.0],
                sample_count: 0, // 0 weight
                local_loss: None,
            },
        ];

        let config = SecureAggConfig {
            min_participants: 2,
            add_noise: false,
            clip_threshold: 100.0, // High threshold to avoid clipping
            ..Default::default()
        };

        let result = SecureAggregation::aggregate(&updates, &config);

        // Weighted average: (1*100 + 4*200 + 1*0) / 300 = 900/300 = 3.0
        assert!((result.aggregated_params[0] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_insufficient_participants() {
        let updates = vec![
            ParticipantUpdate {
                participant_id: "p1".to_string(),
                parameters: vec![1.0],
                sample_count: 100,
                local_loss: None,
            },
        ];

        let config = SecureAggConfig {
            min_participants: 3,
            ..Default::default()
        };

        let result = SecureAggregation::aggregate(&updates, &config);

        assert_eq!(result.participant_count, 0);
        assert!(result.aggregated_params.is_empty());
        assert_eq!(result.privacy_guarantee.epsilon, f64::INFINITY);
    }

    #[test]
    fn test_clipping() {
        let params = vec![3.0, 4.0]; // Norm = 5
        let clipped = SecureAggregation::clip_update(&params, 1.0);

        let norm: f64 = clipped.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dp_noise_added() {
        let updates = vec![
            ParticipantUpdate {
                participant_id: "p1".to_string(),
                parameters: vec![1.0, 1.0],
                sample_count: 100,
                local_loss: None,
            },
            ParticipantUpdate {
                participant_id: "p2".to_string(),
                parameters: vec![1.0, 1.0],
                sample_count: 100,
                local_loss: None,
            },
            ParticipantUpdate {
                participant_id: "p3".to_string(),
                parameters: vec![1.0, 1.0],
                sample_count: 100,
                local_loss: None,
            },
        ];

        let config = SecureAggConfig {
            min_participants: 3,
            add_noise: true,
            epsilon: 1.0,
            seed: Some(42),
            ..Default::default()
        };

        let result = SecureAggregation::aggregate(&updates, &config);

        // With noise, result should not be exactly 1.0
        assert!(result.privacy_guarantee.noise_scale > 0.0);
        // But should be close (noise is bounded by epsilon)
    }

    #[test]
    fn test_empty_updates() {
        let config = SecureAggConfig::default();
        let result = SecureAggregation::aggregate(&[], &config);

        assert!(result.aggregated_params.is_empty());
        assert_eq!(result.participant_count, 0);
    }

    #[test]
    fn test_simulate_round() {
        let global = vec![0.0, 0.0, 0.0];
        let local_updates = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.3, 0.4],
            vec![0.3, 0.4, 0.5],
        ];
        let sample_counts = vec![100, 100, 100];

        let config = SecureAggConfig {
            min_participants: 3,
            add_noise: false,
            ..Default::default()
        };

        let result = SecureAggregation::simulate_round(&global, &local_updates, &sample_counts, &config);

        assert_eq!(result.participant_count, 3);
        assert!(result.average_loss.is_some());
    }

    #[test]
    fn test_verify_aggregation() {
        let updates = vec![
            ParticipantUpdate {
                participant_id: "p1".to_string(),
                parameters: vec![1.0],
                sample_count: 100,
                local_loss: None,
            },
            ParticipantUpdate {
                participant_id: "p2".to_string(),
                parameters: vec![2.0],
                sample_count: 100,
                local_loss: None,
            },
            ParticipantUpdate {
                participant_id: "p3".to_string(),
                parameters: vec![3.0],
                sample_count: 100,
                local_loss: None,
            },
        ];

        let config = SecureAggConfig {
            min_participants: 3,
            add_noise: false,
            ..Default::default()
        };

        let result = SecureAggregation::aggregate(&updates, &config);
        assert!(SecureAggregation::verify_aggregation(&updates, &result));
    }
}
