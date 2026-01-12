//! Forensic query execution kernels.
//!
//! This module provides forensic analysis capabilities:
//! - Historical pattern search
//! - Timeline reconstruction
//! - Activity summarization
//! - Anomaly hunting

use crate::types::{ForensicQuery, ForensicResult, QueryType, UserEvent};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Forensic Query Execution Kernel
// ============================================================================

/// Forensic query execution kernel.
///
/// Executes forensic queries against historical event data for
/// investigation and analysis purposes.
#[derive(Debug, Clone)]
pub struct ForensicQueryExecution {
    metadata: KernelMetadata,
}

impl Default for ForensicQueryExecution {
    fn default() -> Self {
        Self::new()
    }
}

impl ForensicQueryExecution {
    /// Create a new forensic query execution kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch(
                "behavioral/forensic-query",
                Domain::BehavioralAnalytics,
            )
            .with_description("Forensic query execution for historical analysis")
            .with_throughput(5_000)
            .with_latency_us(1000.0),
        }
    }

    /// Execute a forensic query against events.
    ///
    /// # Arguments
    /// * `query` - The forensic query to execute
    /// * `events` - Events to search
    pub fn compute(query: &ForensicQuery, events: &[UserEvent]) -> ForensicResult {
        let start_time = std::time::Instant::now();

        // Apply time filter
        let filtered_events: Vec<_> = events
            .iter()
            .filter(|e| e.timestamp >= query.start_time && e.timestamp <= query.end_time)
            .collect();

        // Apply user filter
        let filtered_events: Vec<_> = if let Some(ref user_ids) = query.user_ids {
            filtered_events
                .into_iter()
                .filter(|e| user_ids.contains(&e.user_id))
                .collect()
        } else {
            filtered_events
        };

        // Apply event type filter
        let filtered_events: Vec<_> = if let Some(ref event_types) = query.event_types {
            filtered_events
                .into_iter()
                .filter(|e| event_types.contains(&e.event_type))
                .collect()
        } else {
            filtered_events
        };

        // Execute query based on type
        let (event_ids, summary) = match query.query_type {
            QueryType::PatternSearch => Self::pattern_search(&filtered_events, &query.filters),
            QueryType::Timeline => Self::timeline_reconstruction(&filtered_events),
            QueryType::ActivitySummary => Self::activity_summary(&filtered_events),
            QueryType::AnomalyHunt => Self::anomaly_hunt(&filtered_events, &query.filters),
            QueryType::Correlation => Self::correlation_analysis(&filtered_events),
        };

        let execution_time_ms = start_time.elapsed().as_millis() as u64;

        ForensicResult {
            query_id: query.id,
            events: event_ids.clone(),
            total_matches: event_ids.len() as u64,
            summary,
            execution_time_ms,
        }
    }

    /// Execute multiple queries in batch.
    pub fn compute_batch(
        queries: &[ForensicQuery],
        events: &[UserEvent],
    ) -> Vec<ForensicResult> {
        queries
            .iter()
            .map(|q| Self::compute(q, events))
            .collect()
    }

    /// Pattern search query.
    fn pattern_search(
        events: &[&UserEvent],
        filters: &HashMap<String, String>,
    ) -> (Vec<u64>, HashMap<String, f64>) {
        let mut matched_ids = Vec::new();
        let mut summary = HashMap::new();

        // Apply custom filters
        for event in events {
            let mut matches = true;

            for (key, expected) in filters {
                match key.as_str() {
                    "event_type_pattern" => {
                        if !event.event_type.contains(expected) {
                            matches = false;
                        }
                    }
                    "device_id" => {
                        if event.device_id.as_ref() != Some(expected) {
                            matches = false;
                        }
                    }
                    "location" => {
                        if event.location.as_ref() != Some(expected) {
                            matches = false;
                        }
                    }
                    "ip_pattern" => {
                        if let Some(ref ip) = event.ip_address {
                            if !ip.contains(expected) {
                                matches = false;
                            }
                        } else {
                            matches = false;
                        }
                    }
                    _ => {}
                }

                if !matches {
                    break;
                }
            }

            if matches {
                matched_ids.push(event.id);
            }
        }

        summary.insert("match_count".to_string(), matched_ids.len() as f64);
        summary.insert("total_searched".to_string(), events.len() as f64);
        summary.insert(
            "match_rate".to_string(),
            matched_ids.len() as f64 / events.len().max(1) as f64,
        );

        (matched_ids, summary)
    }

    /// Timeline reconstruction query.
    fn timeline_reconstruction(events: &[&UserEvent]) -> (Vec<u64>, HashMap<String, f64>) {
        // Sort by timestamp
        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by_key(|e| e.timestamp);

        let event_ids: Vec<_> = sorted.iter().map(|e| e.id).collect();
        let mut summary = HashMap::new();

        if !sorted.is_empty() {
            let first_ts = sorted.first().unwrap().timestamp;
            let last_ts = sorted.last().unwrap().timestamp;
            let duration = (last_ts - first_ts) as f64;

            summary.insert("timeline_start".to_string(), first_ts as f64);
            summary.insert("timeline_end".to_string(), last_ts as f64);
            summary.insert("duration_seconds".to_string(), duration);
            summary.insert("event_count".to_string(), sorted.len() as f64);
            summary.insert(
                "events_per_hour".to_string(),
                sorted.len() as f64 / (duration / 3600.0).max(1.0),
            );

            // Count unique users
            let unique_users: std::collections::HashSet<_> =
                sorted.iter().map(|e| e.user_id).collect();
            summary.insert("unique_users".to_string(), unique_users.len() as f64);

            // Count unique sessions
            let unique_sessions: std::collections::HashSet<_> = sorted
                .iter()
                .filter_map(|e| e.session_id)
                .collect();
            summary.insert("unique_sessions".to_string(), unique_sessions.len() as f64);
        }

        (event_ids, summary)
    }

    /// Activity summary query.
    fn activity_summary(events: &[&UserEvent]) -> (Vec<u64>, HashMap<String, f64>) {
        let event_ids: Vec<_> = events.iter().map(|e| e.id).collect();
        let mut summary = HashMap::new();

        // Event type distribution
        let mut type_counts: HashMap<&str, u64> = HashMap::new();
        for event in events {
            *type_counts.entry(&event.event_type).or_insert(0) += 1;
        }

        let total = events.len() as f64;
        for (event_type, count) in &type_counts {
            let key = format!("type_{}_count", event_type);
            summary.insert(key, *count as f64);

            let ratio_key = format!("type_{}_ratio", event_type);
            summary.insert(ratio_key, *count as f64 / total);
        }

        // Hourly distribution
        let mut hour_counts = [0u64; 24];
        for event in events {
            let hour = ((event.timestamp / 3600) % 24) as usize;
            hour_counts[hour] += 1;
        }

        let peak_hour = hour_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| *c)
            .map(|(h, _)| h)
            .unwrap_or(0);
        summary.insert("peak_activity_hour".to_string(), peak_hour as f64);

        // User activity
        let mut user_counts: HashMap<u64, u64> = HashMap::new();
        for event in events {
            *user_counts.entry(event.user_id).or_insert(0) += 1;
        }

        summary.insert("unique_users".to_string(), user_counts.len() as f64);

        if !user_counts.is_empty() {
            let avg_events_per_user = total / user_counts.len() as f64;
            summary.insert("avg_events_per_user".to_string(), avg_events_per_user);

            let max_user_events = *user_counts.values().max().unwrap_or(&0);
            summary.insert("max_user_events".to_string(), max_user_events as f64);
        }

        // Location distribution
        let unique_locations: std::collections::HashSet<_> = events
            .iter()
            .filter_map(|e| e.location.as_ref())
            .collect();
        summary.insert("unique_locations".to_string(), unique_locations.len() as f64);

        // Device distribution
        let unique_devices: std::collections::HashSet<_> = events
            .iter()
            .filter_map(|e| e.device_id.as_ref())
            .collect();
        summary.insert("unique_devices".to_string(), unique_devices.len() as f64);

        summary.insert("total_events".to_string(), total);

        (event_ids, summary)
    }

    /// Anomaly hunting query.
    fn anomaly_hunt(
        events: &[&UserEvent],
        filters: &HashMap<String, String>,
    ) -> (Vec<u64>, HashMap<String, f64>) {
        let mut anomalous_ids = Vec::new();
        let mut summary = HashMap::new();

        // Parse thresholds from filters
        let velocity_threshold: f64 = filters
            .get("velocity_threshold")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10.0);

        let _time_anomaly_hours: Vec<u8> = filters
            .get("unusual_hours")
            .map(|h| {
                h.split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_else(|| vec![0, 1, 2, 3, 4, 5]);

        // Calculate baseline statistics per user
        let mut user_stats: HashMap<u64, UserStats> = HashMap::new();

        for event in events {
            user_stats
                .entry(event.user_id)
                .or_default()
                .add_event(event);
        }

        // Identify anomalies
        let mut velocity_anomalies = 0u64;
        let mut time_anomalies = 0u64;
        let mut location_anomalies = 0u64;

        for event in events {
            let stats = user_stats.get(&event.user_id).unwrap();
            let mut is_anomaly = false;

            // Velocity check: events in recent window
            let hour_window = events
                .iter()
                .filter(|e| {
                    e.user_id == event.user_id
                        && e.timestamp <= event.timestamp
                        && e.timestamp > event.timestamp.saturating_sub(3600)
                })
                .count();

            if hour_window as f64 > velocity_threshold {
                is_anomaly = true;
                velocity_anomalies += 1;
            }

            // Time anomaly check
            let hour = ((event.timestamp / 3600) % 24) as u8;
            if hour < 6 {
                is_anomaly = true;
                time_anomalies += 1;
            }

            // Location anomaly (if user typically has few locations)
            if let Some(ref location) = event.location {
                if stats.unique_locations.len() > 1 && stats.unique_locations.len() < 3 {
                    if !stats.location_counts.contains_key(location.as_str()) {
                        is_anomaly = true;
                        location_anomalies += 1;
                    }
                }
            }

            if is_anomaly {
                anomalous_ids.push(event.id);
            }
        }

        summary.insert("total_events".to_string(), events.len() as f64);
        summary.insert("anomalous_events".to_string(), anomalous_ids.len() as f64);
        summary.insert(
            "anomaly_rate".to_string(),
            anomalous_ids.len() as f64 / events.len().max(1) as f64,
        );
        summary.insert("velocity_anomalies".to_string(), velocity_anomalies as f64);
        summary.insert("time_anomalies".to_string(), time_anomalies as f64);
        summary.insert("location_anomalies".to_string(), location_anomalies as f64);

        (anomalous_ids, summary)
    }

    /// Correlation analysis query.
    fn correlation_analysis(events: &[&UserEvent]) -> (Vec<u64>, HashMap<String, f64>) {
        let event_ids: Vec<_> = events.iter().map(|e| e.id).collect();
        let mut summary = HashMap::new();

        if events.len() < 2 {
            summary.insert("correlation_count".to_string(), 0.0);
            return (event_ids, summary);
        }

        // Sort by timestamp
        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by_key(|e| e.timestamp);

        // Calculate event type correlations (consecutive pairs)
        let mut pair_counts: HashMap<(&str, &str), u64> = HashMap::new();
        let mut single_counts: HashMap<&str, u64> = HashMap::new();

        for event in &sorted {
            *single_counts.entry(&event.event_type).or_insert(0) += 1;
        }

        for window in sorted.windows(2) {
            *pair_counts
                .entry((&window[0].event_type, &window[1].event_type))
                .or_insert(0) += 1;
        }

        // Find strongest correlations
        let total_pairs = (sorted.len() - 1) as f64;
        let mut correlations: Vec<_> = pair_counts
            .iter()
            .map(|((a, b), &count)| {
                let a_count = single_counts.get(a).copied().unwrap_or(1) as f64;
                let b_count = single_counts.get(b).copied().unwrap_or(1) as f64;

                // Lift = P(A->B) / (P(A) * P(B))
                let p_ab = count as f64 / total_pairs;
                let p_a = a_count / sorted.len() as f64;
                let p_b = b_count / sorted.len() as f64;
                let lift = p_ab / (p_a * p_b);

                (format!("{}->{}", a, b), lift, count)
            })
            .collect();

        correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Add top correlations to summary
        for (i, (pair, lift, count)) in correlations.iter().take(5).enumerate() {
            summary.insert(format!("top{}_pair", i + 1), 0.0); // Can't store string, use separate map
            summary.insert(format!("top{}_lift", i + 1), *lift);
            summary.insert(format!("top{}_count", i + 1), *count as f64);
            // Store pair info in a way that's accessible
            let _ = pair; // Used for display purposes
        }

        summary.insert("unique_event_types".to_string(), single_counts.len() as f64);
        summary.insert("unique_pairs".to_string(), pair_counts.len() as f64);
        summary.insert("total_transitions".to_string(), total_pairs);

        (event_ids, summary)
    }

    /// Create a pattern search query.
    pub fn pattern_search_query(
        id: u64,
        start_time: u64,
        end_time: u64,
        pattern: &str,
    ) -> ForensicQuery {
        let mut filters = HashMap::new();
        filters.insert("event_type_pattern".to_string(), pattern.to_string());

        ForensicQuery {
            id,
            query_type: QueryType::PatternSearch,
            start_time,
            end_time,
            user_ids: None,
            event_types: None,
            filters,
        }
    }

    /// Create a timeline query.
    pub fn timeline_query(
        id: u64,
        start_time: u64,
        end_time: u64,
        user_ids: Option<Vec<u64>>,
    ) -> ForensicQuery {
        ForensicQuery {
            id,
            query_type: QueryType::Timeline,
            start_time,
            end_time,
            user_ids,
            event_types: None,
            filters: HashMap::new(),
        }
    }
}

impl GpuKernel for ForensicQueryExecution {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// User statistics for anomaly detection.
#[derive(Debug, Default)]
struct UserStats {
    event_count: u64,
    unique_locations: std::collections::HashSet<String>,
    location_counts: HashMap<String, u64>,
    unique_devices: std::collections::HashSet<String>,
}

impl UserStats {
    fn add_event(&mut self, event: &UserEvent) {
        self.event_count += 1;

        if let Some(ref loc) = event.location {
            self.unique_locations.insert(loc.clone());
            *self.location_counts.entry(loc.clone()).or_insert(0) += 1;
        }

        if let Some(ref dev) = event.device_id {
            self.unique_devices.insert(dev.clone());
        }
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
                timestamp: base_ts + 60,
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
                timestamp: base_ts + 120,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("device_a".to_string()),
                ip_address: Some("192.168.1.1".to_string()),
                location: Some("US".to_string()),
            },
            UserEvent {
                id: 4,
                user_id: 200,
                event_type: "login".to_string(),
                timestamp: base_ts + 30,
                attributes: HashMap::new(),
                session_id: Some(2),
                device_id: Some("device_b".to_string()),
                ip_address: Some("10.0.0.1".to_string()),
                location: Some("UK".to_string()),
            },
            UserEvent {
                id: 5,
                user_id: 200,
                event_type: "logout".to_string(),
                timestamp: base_ts + 180,
                attributes: HashMap::new(),
                session_id: Some(2),
                device_id: Some("device_b".to_string()),
                ip_address: Some("10.0.0.1".to_string()),
                location: Some("UK".to_string()),
            },
        ]
    }

    #[test]
    fn test_forensic_query_metadata() {
        let kernel = ForensicQueryExecution::new();
        assert_eq!(kernel.metadata().id, "behavioral/forensic-query");
        assert_eq!(kernel.metadata().domain, Domain::BehavioralAnalytics);
    }

    #[test]
    fn test_pattern_search() {
        let events = create_test_events();
        let query = ForensicQueryExecution::pattern_search_query(
            1,
            1700000000,
            1700000500,
            "login",
        );

        let result = ForensicQueryExecution::compute(&query, &events);

        assert_eq!(result.query_id, 1);
        assert!(result.total_matches > 0);
        assert!(result.summary.contains_key("match_count"));
    }

    #[test]
    fn test_timeline_reconstruction() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 2,
            query_type: QueryType::Timeline,
            start_time: 1700000000,
            end_time: 1700000500,
            user_ids: Some(vec![100]),
            event_types: None,
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        assert_eq!(result.query_id, 2);
        assert_eq!(result.total_matches, 3); // User 100 has 3 events
        assert!(result.summary.contains_key("duration_seconds"));
    }

    #[test]
    fn test_activity_summary() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 3,
            query_type: QueryType::ActivitySummary,
            start_time: 1700000000,
            end_time: 1700000500,
            user_ids: None,
            event_types: None,
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        assert!(result.summary.contains_key("unique_users"));
        assert!(result.summary.contains_key("total_events"));
        assert_eq!(result.summary.get("unique_users").copied(), Some(2.0));
    }

    #[test]
    fn test_anomaly_hunt() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 4,
            query_type: QueryType::AnomalyHunt,
            start_time: 1700000000,
            end_time: 1700000500,
            user_ids: None,
            event_types: None,
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        assert!(result.summary.contains_key("anomaly_rate"));
        assert!(result.summary.contains_key("velocity_anomalies"));
    }

    #[test]
    fn test_correlation_analysis() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 5,
            query_type: QueryType::Correlation,
            start_time: 1700000000,
            end_time: 1700000500,
            user_ids: None,
            event_types: None,
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        assert!(result.summary.contains_key("unique_pairs"));
        assert!(result.summary.contains_key("total_transitions"));
    }

    #[test]
    fn test_user_filter() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 6,
            query_type: QueryType::ActivitySummary,
            start_time: 1700000000,
            end_time: 1700000500,
            user_ids: Some(vec![100]),
            event_types: None,
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        assert_eq!(result.summary.get("unique_users").copied(), Some(1.0));
    }

    #[test]
    fn test_event_type_filter() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 7,
            query_type: QueryType::ActivitySummary,
            start_time: 1700000000,
            end_time: 1700000500,
            user_ids: None,
            event_types: Some(vec!["login".to_string()]),
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        assert_eq!(result.total_matches, 2); // 2 login events
    }

    #[test]
    fn test_time_filter() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 8,
            query_type: QueryType::Timeline,
            start_time: 1700000050, // After first event
            end_time: 1700000130,   // Before last events
            user_ids: None,
            event_types: None,
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        // Should only include events in time range
        assert!(result.total_matches < 5);
    }

    #[test]
    fn test_batch_queries() {
        let events = create_test_events();
        let queries = vec![
            ForensicQuery {
                id: 1,
                query_type: QueryType::ActivitySummary,
                start_time: 1700000000,
                end_time: 1700000500,
                user_ids: None,
                event_types: None,
                filters: HashMap::new(),
            },
            ForensicQuery {
                id: 2,
                query_type: QueryType::Timeline,
                start_time: 1700000000,
                end_time: 1700000500,
                user_ids: Some(vec![100]),
                event_types: None,
                filters: HashMap::new(),
            },
        ];

        let results = ForensicQueryExecution::compute_batch(&queries, &events);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].query_id, 1);
        assert_eq!(results[1].query_id, 2);
    }

    #[test]
    fn test_execution_time_tracking() {
        let events = create_test_events();
        let query = ForensicQuery {
            id: 1,
            query_type: QueryType::ActivitySummary,
            start_time: 1700000000,
            end_time: 1700000500,
            user_ids: None,
            event_types: None,
            filters: HashMap::new(),
        };

        let result = ForensicQueryExecution::compute(&query, &events);

        // Execution time should be recorded
        assert!(result.execution_time_ms < 1000); // Should be fast
    }
}
