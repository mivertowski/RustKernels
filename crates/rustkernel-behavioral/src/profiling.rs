//! Behavioral profiling kernels.
//!
//! This module provides user behavioral profiling:
//! - Feature extraction from event streams
//! - Behavioral baseline construction
//! - Deviation scoring from baseline

use crate::types::{
    AnomalyResult, AnomalyType, BehaviorProfile, FeatureDeviation, ProfilingResult, UserEvent,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Behavioral Profiling Kernel
// ============================================================================

/// Behavioral profiling kernel.
///
/// Extracts behavioral features from user event streams to build
/// a behavioral baseline profile.
#[derive(Debug, Clone)]
pub struct BehavioralProfiling {
    metadata: KernelMetadata,
}

impl Default for BehavioralProfiling {
    fn default() -> Self {
        Self::new()
    }
}

impl BehavioralProfiling {
    /// Create a new behavioral profiling kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring(
                "behavioral/profiling",
                Domain::BehavioralAnalytics,
            )
            .with_description("Behavioral feature extraction and profiling")
            .with_throughput(100_000)
            .with_latency_us(50.0),
        }
    }

    /// Extract behavioral profile from user events.
    ///
    /// # Arguments
    /// * `user_id` - The user ID to profile
    /// * `events` - Historical events for the user
    /// * `feature_config` - Feature extraction configuration
    pub fn compute(
        user_id: u64,
        events: &[UserEvent],
        feature_config: &FeatureConfig,
    ) -> ProfilingResult {
        if events.is_empty() {
            return ProfilingResult {
                user_id,
                features: Vec::new(),
                stability: 0.0,
                confidence: 0.0,
            };
        }

        let mut features = Vec::new();

        // Temporal features
        if feature_config.extract_temporal {
            let temporal = Self::extract_temporal_features(events);
            features.extend(temporal);
        }

        // Frequency features
        if feature_config.extract_frequency {
            let frequency = Self::extract_frequency_features(events);
            features.extend(frequency);
        }

        // Session features
        if feature_config.extract_session {
            let session = Self::extract_session_features(events);
            features.extend(session);
        }

        // Device/location features
        if feature_config.extract_device_location {
            let device_loc = Self::extract_device_location_features(events);
            features.extend(device_loc);
        }

        // Calculate profile stability (based on feature variance over time)
        let stability = Self::calculate_stability(events, &features);

        // Calculate confidence (based on event count and time span)
        let confidence = Self::calculate_confidence(events);

        ProfilingResult {
            user_id,
            features,
            stability,
            confidence,
        }
    }

    /// Build a full behavior profile from profiling result.
    pub fn build_profile(result: &ProfilingResult, timestamp: u64) -> BehaviorProfile {
        let feature_names: Vec<String> = result.features.iter().map(|(n, _)| n.clone()).collect();
        let feature_values: Vec<f64> = result.features.iter().map(|(_, v)| *v).collect();

        BehaviorProfile {
            user_id: result.user_id,
            features: feature_values,
            feature_names,
            created_at: timestamp,
            updated_at: timestamp,
            event_count: 0, // Would be set from events.len()
        }
    }

    /// Extract temporal features (time-of-day patterns).
    fn extract_temporal_features(events: &[UserEvent]) -> Vec<(String, f64)> {
        let mut features = Vec::new();

        // Hour distribution
        let mut hour_counts = [0u32; 24];
        for event in events {
            let hour = ((event.timestamp / 3600) % 24) as usize;
            hour_counts[hour] += 1;
        }

        let total = events.len() as f64;

        // Peak hour
        let peak_hour = hour_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| *c)
            .map(|(h, _)| h)
            .unwrap_or(0);
        features.push(("peak_hour".to_string(), peak_hour as f64));

        // Business hours ratio (9-17)
        let business_hours: u32 = hour_counts[9..18].iter().sum();
        features.push((
            "business_hours_ratio".to_string(),
            business_hours as f64 / total,
        ));

        // Night activity ratio (22-6)
        let night_hours: u32 = hour_counts[22..24].iter().sum::<u32>()
            + hour_counts[0..6].iter().sum::<u32>();
        features.push(("night_activity_ratio".to_string(), night_hours as f64 / total));

        // Weekend ratio
        let weekend_events = events
            .iter()
            .filter(|e| {
                let day = (e.timestamp / 86400) % 7;
                day == 5 || day == 6 // Sat, Sun
            })
            .count();
        features.push(("weekend_ratio".to_string(), weekend_events as f64 / total));

        // Hour entropy (measure of temporal spread)
        let hour_entropy = Self::calculate_entropy(&hour_counts);
        features.push(("hour_entropy".to_string(), hour_entropy));

        features
    }

    /// Extract frequency features.
    fn extract_frequency_features(events: &[UserEvent]) -> Vec<(String, f64)> {
        let mut features = Vec::new();

        if events.len() < 2 {
            features.push(("avg_events_per_day".to_string(), 0.0));
            features.push(("event_rate_variance".to_string(), 0.0));
            return features;
        }

        // Time span in days
        let min_ts = events.iter().map(|e| e.timestamp).min().unwrap();
        let max_ts = events.iter().map(|e| e.timestamp).max().unwrap();
        let span_days = ((max_ts - min_ts) as f64 / 86400.0).max(1.0);

        // Average events per day
        let avg_per_day = events.len() as f64 / span_days;
        features.push(("avg_events_per_day".to_string(), avg_per_day));

        // Event type distribution
        let mut type_counts: HashMap<&str, u32> = HashMap::new();
        for event in events {
            *type_counts.entry(&event.event_type).or_insert(0) += 1;
        }

        // Most common event type ratio
        let max_type_count = type_counts.values().max().copied().unwrap_or(0);
        features.push((
            "dominant_event_ratio".to_string(),
            max_type_count as f64 / events.len() as f64,
        ));

        // Event type diversity (unique types / total events)
        features.push((
            "event_type_diversity".to_string(),
            type_counts.len() as f64 / events.len() as f64,
        ));

        // Inter-event time statistics
        let mut inter_times: Vec<f64> = Vec::new();
        let mut sorted_events: Vec<_> = events.iter().collect();
        sorted_events.sort_by_key(|e| e.timestamp);

        for window in sorted_events.windows(2) {
            inter_times.push((window[1].timestamp - window[0].timestamp) as f64);
        }

        if !inter_times.is_empty() {
            let mean_inter = inter_times.iter().sum::<f64>() / inter_times.len() as f64;
            features.push(("mean_inter_event_time".to_string(), mean_inter));

            let variance = inter_times
                .iter()
                .map(|t| (t - mean_inter).powi(2))
                .sum::<f64>()
                / inter_times.len() as f64;
            features.push(("inter_event_variance".to_string(), variance.sqrt()));
        }

        features
    }

    /// Extract session features.
    fn extract_session_features(events: &[UserEvent]) -> Vec<(String, f64)> {
        let mut features = Vec::new();

        // Group events by session
        let mut sessions: HashMap<u64, Vec<&UserEvent>> = HashMap::new();
        let mut no_session_count = 0;

        for event in events {
            if let Some(session_id) = event.session_id {
                sessions.entry(session_id).or_default().push(event);
            } else {
                no_session_count += 1;
            }
        }

        let session_count = sessions.len();
        features.push(("session_count".to_string(), session_count as f64));

        if session_count > 0 {
            // Average events per session
            let avg_events_per_session =
                events.len() as f64 / (session_count + (no_session_count > 0) as usize) as f64;
            features.push(("avg_events_per_session".to_string(), avg_events_per_session));

            // Average session duration
            let session_durations: Vec<f64> = sessions
                .values()
                .map(|session_events| {
                    let min_ts = session_events.iter().map(|e| e.timestamp).min().unwrap();
                    let max_ts = session_events.iter().map(|e| e.timestamp).max().unwrap();
                    (max_ts - min_ts) as f64
                })
                .collect();

            let avg_duration =
                session_durations.iter().sum::<f64>() / session_durations.len() as f64;
            features.push(("avg_session_duration".to_string(), avg_duration));
        } else {
            features.push(("avg_events_per_session".to_string(), 0.0));
            features.push(("avg_session_duration".to_string(), 0.0));
        }

        features
    }

    /// Extract device and location features.
    fn extract_device_location_features(events: &[UserEvent]) -> Vec<(String, f64)> {
        let mut features = Vec::new();

        // Count unique devices
        let unique_devices: std::collections::HashSet<_> = events
            .iter()
            .filter_map(|e| e.device_id.as_ref())
            .collect();
        features.push(("unique_device_count".to_string(), unique_devices.len() as f64));

        // Count unique locations
        let unique_locations: std::collections::HashSet<_> =
            events.iter().filter_map(|e| e.location.as_ref()).collect();
        features.push((
            "unique_location_count".to_string(),
            unique_locations.len() as f64,
        ));

        // Device switching frequency
        let device_switches = Self::count_switches(
            &events
                .iter()
                .filter_map(|e| e.device_id.as_ref().map(|d| d.as_str()))
                .collect::<Vec<_>>(),
        );
        features.push((
            "device_switch_rate".to_string(),
            device_switches as f64 / events.len().max(1) as f64,
        ));

        // Location switching frequency
        let location_switches = Self::count_switches(
            &events
                .iter()
                .filter_map(|e| e.location.as_ref().map(|l| l.as_str()))
                .collect::<Vec<_>>(),
        );
        features.push((
            "location_switch_rate".to_string(),
            location_switches as f64 / events.len().max(1) as f64,
        ));

        features
    }

    /// Count switches in a sequence.
    fn count_switches(sequence: &[&str]) -> usize {
        if sequence.len() < 2 {
            return 0;
        }
        sequence.windows(2).filter(|w| w[0] != w[1]).count()
    }

    /// Calculate entropy of a distribution.
    fn calculate_entropy(counts: &[u32]) -> f64 {
        let total: u32 = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in counts {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Calculate profile stability (0-1, higher is more stable).
    fn calculate_stability(events: &[UserEvent], features: &[(String, f64)]) -> f64 {
        if events.len() < 10 || features.is_empty() {
            return 0.0;
        }

        // Split events into two halves and compare feature stability
        let mid = events.len() / 2;
        let config = FeatureConfig::default();

        let first_half = Self::compute(0, &events[..mid], &config);
        let second_half = Self::compute(0, &events[mid..], &config);

        // Calculate correlation between first and second half features
        let first_map: HashMap<_, _> = first_half.features.into_iter().collect();
        let second_map: HashMap<_, _> = second_half.features.into_iter().collect();

        let mut correlations = Vec::new();
        for (name, v1) in &first_map {
            if let Some(&v2) = second_map.get(name) {
                if v1.abs() > 0.001 || v2.abs() > 0.001 {
                    let similarity = 1.0 - (v1 - v2).abs() / (v1.abs() + v2.abs() + 0.001);
                    correlations.push(similarity);
                }
            }
        }

        if correlations.is_empty() {
            return 0.5;
        }

        correlations.iter().sum::<f64>() / correlations.len() as f64
    }

    /// Calculate confidence in profile (0-1).
    fn calculate_confidence(events: &[UserEvent]) -> f64 {
        if events.is_empty() {
            return 0.0;
        }

        // Event count contribution (more events = higher confidence)
        let count_factor = (events.len() as f64 / 100.0).min(1.0);

        // Time span contribution (longer history = higher confidence)
        let min_ts = events.iter().map(|e| e.timestamp).min().unwrap();
        let max_ts = events.iter().map(|e| e.timestamp).max().unwrap();
        let span_days = (max_ts - min_ts) as f64 / 86400.0;
        let span_factor = (span_days / 30.0).min(1.0); // Max confidence at 30 days

        // Event density contribution
        let density = events.len() as f64 / span_days.max(1.0);
        let density_factor = (density / 10.0).min(1.0); // Max at 10 events/day

        count_factor * 0.4 + span_factor * 0.3 + density_factor * 0.3
    }
}

impl GpuKernel for BehavioralProfiling {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Feature extraction configuration.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Extract temporal features.
    pub extract_temporal: bool,
    /// Extract frequency features.
    pub extract_frequency: bool,
    /// Extract session features.
    pub extract_session: bool,
    /// Extract device/location features.
    pub extract_device_location: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            extract_temporal: true,
            extract_frequency: true,
            extract_session: true,
            extract_device_location: true,
        }
    }
}

// ============================================================================
// Anomaly Profiling Kernel
// ============================================================================

/// Anomaly profiling kernel.
///
/// Compares current event against behavioral baseline to detect anomalies.
#[derive(Debug, Clone)]
pub struct AnomalyProfiling {
    metadata: KernelMetadata,
}

impl Default for AnomalyProfiling {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyProfiling {
    /// Create a new anomaly profiling kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("behavioral/anomaly", Domain::BehavioralAnalytics)
                .with_description("Behavioral anomaly detection")
                .with_throughput(200_000)
                .with_latency_us(25.0),
        }
    }

    /// Detect anomalies by comparing event against profile.
    ///
    /// # Arguments
    /// * `event` - The event to analyze
    /// * `profile` - The user's behavioral profile
    /// * `recent_events` - Recent events for context
    /// * `threshold` - Anomaly score threshold (0-100)
    pub fn compute(
        event: &UserEvent,
        profile: &BehaviorProfile,
        recent_events: &[UserEvent],
        threshold: f64,
    ) -> AnomalyResult {
        let mut deviations = Vec::new();
        let mut total_score: f64 = 0.0;
        let mut anomaly_types = Vec::new();

        // Check temporal anomaly
        let hour = ((event.timestamp / 3600) % 24) as f64;
        if let Some(expected_hour) = profile.get_feature("peak_hour") {
            let hour_diff = (hour - expected_hour).abs().min(24.0 - (hour - expected_hour).abs());
            let hour_score = (hour_diff / 12.0) * 100.0;

            if hour_score > 30.0 {
                deviations.push(FeatureDeviation {
                    feature_name: "hour".to_string(),
                    expected: expected_hour,
                    actual: hour,
                    z_score: hour_diff / 6.0,
                    contribution: hour_score * 0.2,
                });
                anomaly_types.push(AnomalyType::Temporal);
            }
            total_score += hour_score * 0.2;
        }

        // Check location anomaly
        if let Some(location) = &event.location {
            if let Some(unique_locs) = profile.get_feature("unique_location_count") {
                // If user typically has few locations, new location is more suspicious
                if unique_locs < 3.0 {
                    // Check if this is a new location by comparing with recent events
                    let known_locations: std::collections::HashSet<_> = recent_events
                        .iter()
                        .filter_map(|e| e.location.as_ref())
                        .collect();

                    if !known_locations.contains(location) {
                        let geo_score = 50.0;
                        deviations.push(FeatureDeviation {
                            feature_name: "location".to_string(),
                            expected: unique_locs,
                            actual: unique_locs + 1.0,
                            z_score: 2.0,
                            contribution: geo_score * 0.25,
                        });
                        anomaly_types.push(AnomalyType::Geographic);
                        total_score += geo_score * 0.25;
                    }
                }
            }
        }

        // Check device anomaly
        if let Some(device) = &event.device_id {
            if let Some(unique_devices) = profile.get_feature("unique_device_count") {
                if unique_devices < 3.0 {
                    let known_devices: std::collections::HashSet<_> = recent_events
                        .iter()
                        .filter_map(|e| e.device_id.as_ref())
                        .collect();

                    if !known_devices.contains(device) {
                        let device_score = 40.0;
                        deviations.push(FeatureDeviation {
                            feature_name: "device".to_string(),
                            expected: unique_devices,
                            actual: unique_devices + 1.0,
                            z_score: 1.5,
                            contribution: device_score * 0.2,
                        });
                        anomaly_types.push(AnomalyType::Device);
                        total_score += device_score * 0.2;
                    }
                }
            }
        }

        // Check velocity anomaly (events in recent window)
        let window_start = event.timestamp.saturating_sub(3600); // 1 hour window
        let recent_count = recent_events
            .iter()
            .filter(|e| e.timestamp >= window_start && e.timestamp <= event.timestamp)
            .count();

        if let Some(avg_per_day) = profile.get_feature("avg_events_per_day") {
            let expected_per_hour = avg_per_day / 24.0;
            if recent_count as f64 > expected_per_hour * 5.0 {
                let velocity_score = ((recent_count as f64 / expected_per_hour) - 1.0).min(100.0);
                deviations.push(FeatureDeviation {
                    feature_name: "velocity".to_string(),
                    expected: expected_per_hour,
                    actual: recent_count as f64,
                    z_score: (recent_count as f64 - expected_per_hour) / expected_per_hour.max(1.0),
                    contribution: velocity_score * 0.35,
                });
                anomaly_types.push(AnomalyType::Velocity);
                total_score += velocity_score * 0.35;
            }
        }

        // Determine overall anomaly type
        let anomaly_type = if anomaly_types.is_empty() {
            None
        } else if anomaly_types.len() > 1 {
            Some(AnomalyType::Mixed)
        } else {
            Some(anomaly_types[0])
        };

        AnomalyResult {
            user_id: event.user_id,
            event_id: event.id,
            anomaly_score: total_score.min(100.0),
            is_anomaly: total_score >= threshold,
            deviations,
            anomaly_type,
        }
    }

    /// Batch analyze multiple events.
    pub fn compute_batch(
        events: &[UserEvent],
        profile: &BehaviorProfile,
        threshold: f64,
    ) -> Vec<AnomalyResult> {
        let mut results = Vec::new();

        for (i, event) in events.iter().enumerate() {
            // Use preceding events as context
            let recent = &events[..i];
            let result = Self::compute(event, profile, recent, threshold);
            results.push(result);
        }

        results
    }
}

impl GpuKernel for AnomalyProfiling {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_events() -> Vec<UserEvent> {
        let base_ts = 1700000000u64;
        vec![
            UserEvent {
                id: 1,
                user_id: 100,
                event_type: "login".to_string(),
                timestamp: base_ts + 36000, // 10:00
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
                timestamp: base_ts + 36300, // 10:05
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
                timestamp: base_ts + 37800, // 10:30
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("device_a".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                location: Some("US".to_string()),
            },
            UserEvent {
                id: 4,
                user_id: 100,
                event_type: "logout".to_string(),
                timestamp: base_ts + 39600, // 11:00
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("device_a".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                location: Some("US".to_string()),
            },
        ]
    }

    #[test]
    fn test_behavioral_profiling_metadata() {
        let kernel = BehavioralProfiling::new();
        assert_eq!(kernel.metadata().id, "behavioral/profiling");
        assert_eq!(kernel.metadata().domain, Domain::BehavioralAnalytics);
    }

    #[test]
    fn test_feature_extraction() {
        let events = create_test_events();
        let config = FeatureConfig::default();

        let result = BehavioralProfiling::compute(100, &events, &config);

        assert_eq!(result.user_id, 100);
        assert!(!result.features.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_temporal_features() {
        let events = create_test_events();
        let config = FeatureConfig {
            extract_temporal: true,
            extract_frequency: false,
            extract_session: false,
            extract_device_location: false,
        };

        let result = BehavioralProfiling::compute(100, &events, &config);

        let feature_map: HashMap<_, _> = result.features.into_iter().collect();
        assert!(feature_map.contains_key("peak_hour"));
        assert!(feature_map.contains_key("business_hours_ratio"));
    }

    #[test]
    fn test_empty_events() {
        let config = FeatureConfig::default();
        let result = BehavioralProfiling::compute(100, &[], &config);

        assert_eq!(result.user_id, 100);
        assert!(result.features.is_empty());
        assert_eq!(result.stability, 0.0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_build_profile() {
        let events = create_test_events();
        let config = FeatureConfig::default();
        let result = BehavioralProfiling::compute(100, &events, &config);

        let profile = BehavioralProfiling::build_profile(&result, 1700000000);

        assert_eq!(profile.user_id, 100);
        assert_eq!(profile.features.len(), profile.feature_names.len());
    }

    #[test]
    fn test_anomaly_profiling_metadata() {
        let kernel = AnomalyProfiling::new();
        assert_eq!(kernel.metadata().id, "behavioral/anomaly");
    }

    #[test]
    fn test_anomaly_detection_normal() {
        let events = create_test_events();
        let config = FeatureConfig::default();
        let result = BehavioralProfiling::compute(100, &events, &config);
        let profile = BehavioralProfiling::build_profile(&result, 1700000000);

        // Create a normal event (same time, same device, same location)
        let normal_event = UserEvent {
            id: 5,
            user_id: 100,
            event_type: "view".to_string(),
            timestamp: 1700000000 + 36000, // Same hour as profile
            attributes: HashMap::new(),
            session_id: Some(2),
            device_id: Some("device_a".to_string()),
            ip_address: Some("192.168.1.1".to_string()),
            location: Some("US".to_string()),
        };

        let anomaly = AnomalyProfiling::compute(&normal_event, &profile, &events, 50.0);

        assert_eq!(anomaly.user_id, 100);
        // Normal event should have low anomaly score
        assert!(anomaly.anomaly_score < 50.0);
    }

    #[test]
    fn test_anomaly_detection_new_device() {
        let events = create_test_events();
        let config = FeatureConfig::default();
        let result = BehavioralProfiling::compute(100, &events, &config);
        let profile = BehavioralProfiling::build_profile(&result, 1700000000);

        // Create event from new device
        let suspicious_event = UserEvent {
            id: 5,
            user_id: 100,
            event_type: "login".to_string(),
            timestamp: 1700000000 + 36000,
            attributes: HashMap::new(),
            session_id: Some(2),
            device_id: Some("unknown_device".to_string()),
            ip_address: Some("10.0.0.1".to_string()),
            location: Some("US".to_string()),
        };

        let anomaly = AnomalyProfiling::compute(&suspicious_event, &profile, &events, 30.0);

        // Should detect device anomaly
        assert!(
            anomaly.deviations.iter().any(|d| d.feature_name == "device"),
            "Should detect device deviation"
        );
    }

    #[test]
    fn test_batch_analysis() {
        let events = create_test_events();
        let config = FeatureConfig::default();
        let result = BehavioralProfiling::compute(100, &events, &config);
        let profile = BehavioralProfiling::build_profile(&result, 1700000000);

        let new_events: Vec<UserEvent> = (0..5)
            .map(|i| UserEvent {
                id: 10 + i as u64,
                user_id: 100,
                event_type: "view".to_string(),
                timestamp: 1700000000 + 40000 + (i as u64 * 300),
                attributes: HashMap::new(),
                session_id: Some(3),
                device_id: Some("device_a".to_string()),
                ip_address: None,
                location: Some("US".to_string()),
            })
            .collect();

        let results = AnomalyProfiling::compute_batch(&new_events, &profile, 50.0);

        assert_eq!(results.len(), 5);
    }
}
