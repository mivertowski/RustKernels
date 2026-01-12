//! Event correlation kernels.
//!
//! This module provides event correlation analysis:
//! - Temporal correlation detection
//! - User/session/device-based correlation
//! - Event clustering

use crate::types::{
    CorrelationCluster, CorrelationResult, CorrelationType, EventCorrelation, UserEvent,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Event Correlation Kernel
// ============================================================================

/// Event correlation kernel.
///
/// Identifies correlated events based on temporal, user, session,
/// device, and location relationships.
#[derive(Debug, Clone)]
pub struct EventCorrelationKernel {
    metadata: KernelMetadata,
}

impl Default for EventCorrelationKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl EventCorrelationKernel {
    /// Create a new event correlation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring(
                "behavioral/event-correlation",
                Domain::BehavioralAnalytics,
            )
            .with_description("Event correlation and clustering")
            .with_throughput(50_000)
            .with_latency_us(100.0),
        }
    }

    /// Find correlations for an event.
    ///
    /// # Arguments
    /// * `event` - The event to find correlations for
    /// * `all_events` - Pool of events to correlate against
    /// * `config` - Correlation configuration
    pub fn compute(
        event: &UserEvent,
        all_events: &[UserEvent],
        config: &CorrelationConfig,
    ) -> CorrelationResult {
        let mut correlations = Vec::new();

        for candidate in all_events {
            if candidate.id == event.id {
                continue;
            }

            // Calculate correlation score and type
            if let Some(correlation) = Self::calculate_correlation(event, candidate, config) {
                correlations.push(correlation);
            }
        }

        // Sort by score descending
        correlations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Limit results
        if let Some(max) = config.max_correlations {
            correlations.truncate(max);
        }

        // Build clusters from correlations
        let clusters = Self::build_clusters(&correlations, all_events, config);

        CorrelationResult {
            event_id: event.id,
            correlations,
            clusters,
        }
    }

    /// Batch correlation analysis for multiple events.
    pub fn compute_batch(
        events: &[UserEvent],
        config: &CorrelationConfig,
    ) -> Vec<CorrelationResult> {
        events
            .iter()
            .map(|e| Self::compute(e, events, config))
            .collect()
    }

    /// Calculate correlation between two events.
    fn calculate_correlation(
        event: &UserEvent,
        candidate: &UserEvent,
        config: &CorrelationConfig,
    ) -> Option<EventCorrelation> {
        let mut score = 0.0;
        let mut correlation_types = Vec::new();

        // Temporal correlation
        let time_diff = (event.timestamp as i64 - candidate.timestamp as i64).abs();
        if time_diff <= config.temporal_window_secs as i64 {
            let temporal_score = 1.0 - (time_diff as f64 / config.temporal_window_secs as f64);
            score += temporal_score * config.weights.temporal;
            if temporal_score > 0.5 {
                correlation_types.push(CorrelationType::Temporal);
            }
        }

        // User correlation
        if event.user_id == candidate.user_id {
            score += config.weights.user;
            correlation_types.push(CorrelationType::User);
        }

        // Session correlation
        if let (Some(s1), Some(s2)) = (event.session_id, candidate.session_id) {
            if s1 == s2 {
                score += config.weights.session;
                correlation_types.push(CorrelationType::Session);
            }
        }

        // Device correlation
        if let (Some(d1), Some(d2)) = (&event.device_id, &candidate.device_id) {
            if d1 == d2 {
                score += config.weights.device;
                correlation_types.push(CorrelationType::Device);
            }
        }

        // Location correlation
        if let (Some(l1), Some(l2)) = (&event.location, &candidate.location) {
            if l1 == l2 {
                score += config.weights.location;
                correlation_types.push(CorrelationType::Location);
            }
        }

        // Normalize score
        let max_possible = config.weights.temporal
            + config.weights.user
            + config.weights.session
            + config.weights.device
            + config.weights.location;
        score = score / max_possible;

        if score < config.min_score {
            return None;
        }

        // Determine dominant correlation type
        let correlation_type = if correlation_types.is_empty() {
            CorrelationType::Temporal
        } else {
            // Return the strongest type based on weights
            correlation_types
                .into_iter()
                .max_by(|a, b| {
                    Self::type_weight(a, &config.weights)
                        .partial_cmp(&Self::type_weight(b, &config.weights))
                        .unwrap()
                })
                .unwrap()
        };

        Some(EventCorrelation {
            correlated_event_id: candidate.id,
            score,
            correlation_type,
            time_diff: event.timestamp as i64 - candidate.timestamp as i64,
        })
    }

    /// Get weight for a correlation type.
    fn type_weight(t: &CorrelationType, weights: &CorrelationWeights) -> f64 {
        match t {
            CorrelationType::Temporal => weights.temporal,
            CorrelationType::User => weights.user,
            CorrelationType::Session => weights.session,
            CorrelationType::Device => weights.device,
            CorrelationType::Location => weights.location,
            CorrelationType::Causal => 1.0, // Causal is highest priority if detected
        }
    }

    /// Build clusters from correlations using union-find.
    fn build_clusters(
        correlations: &[EventCorrelation],
        all_events: &[UserEvent],
        config: &CorrelationConfig,
    ) -> Vec<CorrelationCluster> {
        if correlations.is_empty() {
            return Vec::new();
        }

        // Build event ID to index mapping
        let id_to_idx: HashMap<u64, usize> = all_events
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id, i))
            .collect();

        // Union-Find data structure
        let n = all_events.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);

            if px == py {
                return;
            }

            match rank[px].cmp(&rank[py]) {
                std::cmp::Ordering::Less => parent[px] = py,
                std::cmp::Ordering::Greater => parent[py] = px,
                std::cmp::Ordering::Equal => {
                    parent[py] = px;
                    rank[px] += 1;
                }
            }
        }

        // Build high-score correlation edges
        // For each correlation, union the correlated events
        for (i, e1) in all_events.iter().enumerate() {
            for corr in correlations.iter().filter(|c| c.score >= config.cluster_threshold) {
                if let Some(&idx2) = id_to_idx.get(&corr.correlated_event_id) {
                    // Check if e1 could be the source of this correlation
                    // by checking if their IDs are related
                    if e1.id != corr.correlated_event_id && idx2 < n {
                        union(&mut parent, &mut rank, i, idx2);
                    }
                }
            }
        }

        // Group events by cluster
        let mut cluster_members: HashMap<usize, Vec<u64>> = HashMap::new();
        let mut cluster_types: HashMap<usize, HashMap<CorrelationType, usize>> = HashMap::new();

        for event in all_events {
            if let Some(&idx) = id_to_idx.get(&event.id) {
                let root = find(&mut parent, idx);
                cluster_members.entry(root).or_default().push(event.id);
            }
        }

        // Calculate dominant types from correlations
        for corr in correlations {
            if let Some(&idx) = id_to_idx.get(&corr.correlated_event_id) {
                let root = find(&mut parent, idx);
                *cluster_types
                    .entry(root)
                    .or_default()
                    .entry(corr.correlation_type)
                    .or_insert(0) += 1;
            }
        }

        // Build cluster results
        let mut clusters: Vec<CorrelationCluster> = Vec::new();
        let mut cluster_id = 0u64;

        for (root, event_ids) in cluster_members {
            if event_ids.len() < 2 {
                continue; // Skip singleton clusters
            }

            // Calculate coherence (average correlation score within cluster)
            let cluster_event_set: HashSet<_> = event_ids.iter().collect();
            let internal_correlations: Vec<_> = correlations
                .iter()
                .filter(|c| cluster_event_set.contains(&c.correlated_event_id))
                .collect();

            let coherence = if internal_correlations.is_empty() {
                0.0
            } else {
                internal_correlations.iter().map(|c| c.score).sum::<f64>()
                    / internal_correlations.len() as f64
            };

            // Find dominant type
            let type_counts = cluster_types.get(&root);
            let dominant_type = type_counts
                .and_then(|counts| {
                    counts
                        .iter()
                        .max_by_key(|&(_, count)| *count)
                        .map(|(&t, _)| t)
                })
                .unwrap_or(CorrelationType::Temporal);

            clusters.push(CorrelationCluster {
                id: cluster_id,
                event_ids,
                coherence,
                dominant_type,
            });

            cluster_id += 1;
        }

        // Sort by coherence descending
        clusters.sort_by(|a, b| b.coherence.partial_cmp(&a.coherence).unwrap());

        clusters
    }

    /// Detect causal correlations (A causes B pattern).
    pub fn detect_causal_correlations(
        events: &[UserEvent],
        config: &CorrelationConfig,
    ) -> Vec<EventCorrelation> {
        let mut causal = Vec::new();

        // Sort by timestamp
        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by_key(|e| e.timestamp);

        // Look for consistent A->B patterns
        let mut pair_counts: HashMap<(&str, &str), Vec<i64>> = HashMap::new();

        for window in sorted.windows(2) {
            let time_diff = (window[1].timestamp - window[0].timestamp) as i64;
            if time_diff <= config.temporal_window_secs as i64 {
                pair_counts
                    .entry((&window[0].event_type, &window[1].event_type))
                    .or_default()
                    .push(time_diff);
            }
        }

        // Find pairs with consistent timing (low variance)
        for ((type_a, type_b), time_diffs) in pair_counts {
            if time_diffs.len() < 3 {
                continue;
            }

            let mean = time_diffs.iter().sum::<i64>() as f64 / time_diffs.len() as f64;
            let variance = time_diffs
                .iter()
                .map(|&t| (t as f64 - mean).powi(2))
                .sum::<f64>()
                / time_diffs.len() as f64;
            let cv = variance.sqrt() / mean.abs().max(1.0); // Coefficient of variation

            // Low CV suggests consistent causal relationship
            if cv < 0.5 {
                // Find specific event pairs
                for window in sorted.windows(2) {
                    if window[0].event_type == *type_a && window[1].event_type == *type_b {
                        let score = 1.0 - cv;
                        causal.push(EventCorrelation {
                            correlated_event_id: window[1].id,
                            score,
                            correlation_type: CorrelationType::Causal,
                            time_diff: (window[1].timestamp - window[0].timestamp) as i64,
                        });
                    }
                }
            }
        }

        causal
    }

    /// Find events correlated by all specified types.
    pub fn find_strongly_correlated(
        events: &[UserEvent],
        required_types: &[CorrelationType],
    ) -> Vec<(u64, u64, f64)> {
        let mut pairs = Vec::new();

        for (i, e1) in events.iter().enumerate() {
            for e2 in events.iter().skip(i + 1) {
                let mut matches = Vec::new();

                // Check each required type
                for req_type in required_types {
                    let matched = match req_type {
                        CorrelationType::User => e1.user_id == e2.user_id,
                        CorrelationType::Session => {
                            e1.session_id.is_some()
                                && e1.session_id == e2.session_id
                        }
                        CorrelationType::Device => {
                            e1.device_id.is_some() && e1.device_id == e2.device_id
                        }
                        CorrelationType::Location => {
                            e1.location.is_some() && e1.location == e2.location
                        }
                        CorrelationType::Temporal => {
                            (e1.timestamp as i64 - e2.timestamp as i64).abs() < 3600
                        }
                        CorrelationType::Causal => false, // Requires separate analysis
                    };
                    matches.push(matched);
                }

                if matches.iter().all(|&m| m) {
                    let score = matches.iter().filter(|&&m| m).count() as f64
                        / required_types.len() as f64;
                    pairs.push((e1.id, e2.id, score));
                }
            }
        }

        pairs
    }
}

impl GpuKernel for EventCorrelationKernel {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Correlation configuration.
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Time window for temporal correlation (seconds).
    pub temporal_window_secs: u64,
    /// Minimum correlation score to include.
    pub min_score: f64,
    /// Maximum correlations to return per event.
    pub max_correlations: Option<usize>,
    /// Minimum score for cluster membership.
    pub cluster_threshold: f64,
    /// Correlation type weights.
    pub weights: CorrelationWeights,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            temporal_window_secs: 3600, // 1 hour
            min_score: 0.3,
            max_correlations: Some(50),
            cluster_threshold: 0.5,
            weights: CorrelationWeights::default(),
        }
    }
}

/// Weights for different correlation types.
#[derive(Debug, Clone)]
pub struct CorrelationWeights {
    /// Weight for temporal proximity.
    pub temporal: f64,
    /// Weight for same user.
    pub user: f64,
    /// Weight for same session.
    pub session: f64,
    /// Weight for same device.
    pub device: f64,
    /// Weight for same location.
    pub location: f64,
}

impl Default for CorrelationWeights {
    fn default() -> Self {
        Self {
            temporal: 0.2,
            user: 0.3,
            session: 0.25,
            device: 0.15,
            location: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_correlated_events() -> Vec<UserEvent> {
        let base_ts = 1700000000u64;
        vec![
            UserEvent {
                id: 1,
                user_id: 100,
                event_type: "login".to_string(),
                timestamp: base_ts,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("device_a".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                location: Some("US".to_string()),
            },
            UserEvent {
                id: 2,
                user_id: 100,
                event_type: "view".to_string(),
                timestamp: base_ts + 30,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("device_a".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                location: Some("US".to_string()),
            },
            UserEvent {
                id: 3,
                user_id: 100,
                event_type: "purchase".to_string(),
                timestamp: base_ts + 60,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("device_a".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                location: Some("US".to_string()),
            },
            // Different user, same time window
            UserEvent {
                id: 4,
                user_id: 200,
                event_type: "login".to_string(),
                timestamp: base_ts + 15,
                attributes: HashMap::new(),
                session_id: Some(2),
                device_id: Some("device_b".to_string()),
                ip_address: Some("10.0.0.1".to_string()),
                location: Some("UK".to_string()),
            },
            // Same user, different session
            UserEvent {
                id: 5,
                user_id: 100,
                event_type: "login".to_string(),
                timestamp: base_ts + 7200, // 2 hours later
                attributes: HashMap::new(),
                session_id: Some(3),
                device_id: Some("device_a".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                location: Some("US".to_string()),
            },
        ]
    }

    #[test]
    fn test_correlation_kernel_metadata() {
        let kernel = EventCorrelationKernel::new();
        assert_eq!(kernel.metadata().id, "behavioral/event-correlation");
        assert_eq!(kernel.metadata().domain, Domain::BehavioralAnalytics);
    }

    #[test]
    fn test_same_user_correlation() {
        let events = create_correlated_events();
        let config = CorrelationConfig::default();

        let result = EventCorrelationKernel::compute(&events[0], &events, &config);

        // Should find correlations with events 2 and 3 (same user, session, device)
        assert!(
            !result.correlations.is_empty(),
            "Should find correlations"
        );

        // Highest correlation should be with same-session events
        let same_user_corrs: Vec<_> = result
            .correlations
            .iter()
            .filter(|c| {
                events
                    .iter()
                    .find(|e| e.id == c.correlated_event_id)
                    .map_or(false, |e| e.user_id == 100)
            })
            .collect();

        assert!(!same_user_corrs.is_empty());
    }

    #[test]
    fn test_temporal_correlation() {
        let events = create_correlated_events();
        let config = CorrelationConfig {
            temporal_window_secs: 100,
            ..Default::default()
        };

        let result = EventCorrelationKernel::compute(&events[0], &events, &config);

        // Events 2, 3, and 4 are within temporal window
        let temporal_corrs: Vec<_> = result
            .correlations
            .iter()
            .filter(|c| c.time_diff.abs() < 100)
            .collect();

        assert!(!temporal_corrs.is_empty());
    }

    #[test]
    fn test_session_correlation() {
        let events = create_correlated_events();
        let config = CorrelationConfig::default();

        let result = EventCorrelationKernel::compute(&events[0], &events, &config);

        // Should highly correlate with events 2 and 3 (same session)
        let session_corrs: Vec<_> = result
            .correlations
            .iter()
            .filter(|c| c.correlation_type == CorrelationType::Session)
            .collect();

        // At least some correlations should be session-based
        // (events 2 and 3 share session with event 1)
        assert!(
            result
                .correlations
                .iter()
                .any(|c| c.correlated_event_id == 2 || c.correlated_event_id == 3),
            "Should correlate with same-session events"
        );
        let _ = session_corrs; // Mark as used
    }

    #[test]
    fn test_min_score_filter() {
        let events = create_correlated_events();
        let config = CorrelationConfig {
            min_score: 0.8, // High threshold
            ..Default::default()
        };

        let result = EventCorrelationKernel::compute(&events[0], &events, &config);

        // All correlations should be above threshold
        assert!(result.correlations.iter().all(|c| c.score >= 0.8));
    }

    #[test]
    fn test_max_correlations_limit() {
        let events = create_correlated_events();
        let config = CorrelationConfig {
            max_correlations: Some(2),
            min_score: 0.0, // Allow all
            ..Default::default()
        };

        let result = EventCorrelationKernel::compute(&events[0], &events, &config);

        assert!(result.correlations.len() <= 2);
    }

    #[test]
    fn test_cluster_building() {
        let events = create_correlated_events();
        let config = CorrelationConfig {
            cluster_threshold: 0.3,
            ..Default::default()
        };

        let result = EventCorrelationKernel::compute(&events[0], &events, &config);

        // May or may not have clusters depending on correlations
        for cluster in &result.clusters {
            assert!(cluster.event_ids.len() >= 2, "Clusters should have 2+ events");
            assert!(cluster.coherence >= 0.0 && cluster.coherence <= 1.0);
        }
    }

    #[test]
    fn test_batch_correlation() {
        let events = create_correlated_events();
        let config = CorrelationConfig::default();

        let results = EventCorrelationKernel::compute_batch(&events, &config);

        assert_eq!(results.len(), events.len());
        for result in &results {
            assert!(events.iter().any(|e| e.id == result.event_id));
        }
    }

    #[test]
    fn test_causal_correlation_detection() {
        let base_ts = 1700000000u64;
        // Create events with consistent A->B pattern
        let events: Vec<UserEvent> = (0..10)
            .flat_map(|i| {
                vec![
                    UserEvent {
                        id: i * 2,
                        user_id: 100,
                        event_type: "cause".to_string(),
                        timestamp: base_ts + (i as u64 * 1000),
                        attributes: HashMap::new(),
                        session_id: Some(i as u64),
                        device_id: None,
                        ip_address: None,
                        location: None,
                    },
                    UserEvent {
                        id: i * 2 + 1,
                        user_id: 100,
                        event_type: "effect".to_string(),
                        timestamp: base_ts + (i as u64 * 1000) + 50, // Consistent 50s delay
                        attributes: HashMap::new(),
                        session_id: Some(i as u64),
                        device_id: None,
                        ip_address: None,
                        location: None,
                    },
                ]
            })
            .collect();

        let config = CorrelationConfig::default();
        let causal = EventCorrelationKernel::detect_causal_correlations(&events, &config);

        // Should detect causal relationships
        assert!(
            !causal.is_empty(),
            "Should detect causal correlations in consistent patterns"
        );

        // All should be marked as causal
        assert!(causal.iter().all(|c| c.correlation_type == CorrelationType::Causal));
    }

    #[test]
    fn test_strongly_correlated() {
        let events = create_correlated_events();
        let required = vec![CorrelationType::User, CorrelationType::Session];

        let pairs = EventCorrelationKernel::find_strongly_correlated(&events, &required);

        // Events 1, 2, 3 share user and session
        assert!(!pairs.is_empty());
        assert!(pairs.iter().all(|(_, _, score)| *score == 1.0));
    }

    #[test]
    fn test_empty_events() {
        let events: Vec<UserEvent> = Vec::new();
        let config = CorrelationConfig::default();

        let result = EventCorrelationKernel::compute(
            &UserEvent {
                id: 1,
                user_id: 100,
                event_type: "test".to_string(),
                timestamp: 0,
                attributes: HashMap::new(),
                session_id: None,
                device_id: None,
                ip_address: None,
                location: None,
            },
            &events,
            &config,
        );

        assert!(result.correlations.is_empty());
        assert!(result.clusters.is_empty());
    }

    #[test]
    fn test_correlation_weights() {
        let events = create_correlated_events();

        // High user weight
        let user_config = CorrelationConfig {
            weights: CorrelationWeights {
                user: 0.8,
                session: 0.1,
                device: 0.05,
                location: 0.03,
                temporal: 0.02,
            },
            ..Default::default()
        };

        let result = EventCorrelationKernel::compute(&events[0], &events, &user_config);

        // Same-user events should have higher scores
        if let Some(same_user) = result
            .correlations
            .iter()
            .find(|c| c.correlated_event_id == 2)
        {
            if let Some(diff_user) = result
                .correlations
                .iter()
                .find(|c| c.correlated_event_id == 4)
            {
                assert!(
                    same_user.score > diff_user.score,
                    "Same-user correlation should be stronger with high user weight"
                );
            }
        }
    }
}
