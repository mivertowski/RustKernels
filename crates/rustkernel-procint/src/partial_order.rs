//! Partial order analysis kernel.
//!
//! This module provides partial order analysis for event logs:
//! - Concurrent activity detection
//! - Sequential relationship extraction
//! - Exclusive activity pair identification
//! - Parallelism score calculation

use crate::types::{EventLog, PartialOrderResult, Trace};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Partial Order Analysis Kernel
// ============================================================================

/// Partial order analysis kernel.
///
/// Analyzes event logs to detect concurrency, sequentiality, and exclusivity
/// between activities based on their occurrence patterns across traces.
#[derive(Debug, Clone)]
pub struct PartialOrderAnalysis {
    metadata: KernelMetadata,
}

impl Default for PartialOrderAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialOrderAnalysis {
    /// Create a new partial order analysis kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("procint/partial-order", Domain::ProcessIntelligence)
                .with_description("Partial order and concurrency analysis")
                .with_throughput(50_000)
                .with_latency_us(100.0),
        }
    }

    /// Analyze partial orders in an event log.
    pub fn analyze(log: &EventLog, config: &PartialOrderConfig) -> PartialOrderResult {
        let activities: Vec<String> = log
            .activities()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        if activities.is_empty() {
            return PartialOrderResult {
                concurrent_pairs: Vec::new(),
                sequential_pairs: Vec::new(),
                exclusive_pairs: Vec::new(),
                parallelism_score: 0.0,
            };
        }

        // Build co-occurrence and ordering matrices
        let mut cooccurrence: HashMap<(String, String), u64> = HashMap::new();
        let mut before_count: HashMap<(String, String), u64> = HashMap::new();
        let mut after_count: HashMap<(String, String), u64> = HashMap::new();

        for trace in log.traces.values() {
            let mut sorted_events: Vec<_> = trace.events.iter().collect();
            sorted_events.sort_by_key(|e| e.timestamp);

            // Activities in this trace
            let trace_activities: HashSet<_> =
                sorted_events.iter().map(|e| e.activity.clone()).collect();

            // Count co-occurrences
            for a1 in &trace_activities {
                for a2 in &trace_activities {
                    if a1 != a2 {
                        let key = (a1.clone(), a2.clone());
                        *cooccurrence.entry(key).or_insert(0) += 1;
                    }
                }
            }

            // Count ordering relationships
            for i in 0..sorted_events.len() {
                for j in (i + 1)..sorted_events.len() {
                    let a1 = &sorted_events[i].activity;
                    let a2 = &sorted_events[j].activity;

                    if a1 != a2 {
                        *before_count.entry((a1.clone(), a2.clone())).or_insert(0) += 1;
                        *after_count.entry((a2.clone(), a1.clone())).or_insert(0) += 1;
                    }
                }
            }
        }

        let trace_count = log.trace_count() as u64;
        let mut concurrent_pairs = Vec::new();
        let mut sequential_pairs = Vec::new();
        let mut exclusive_pairs = Vec::new();

        // Analyze each pair of activities
        for i in 0..activities.len() {
            for j in (i + 1)..activities.len() {
                let a1 = &activities[i];
                let a2 = &activities[j];

                let co = cooccurrence
                    .get(&(a1.clone(), a2.clone()))
                    .copied()
                    .unwrap_or(0);
                let ab = before_count
                    .get(&(a1.clone(), a2.clone()))
                    .copied()
                    .unwrap_or(0);
                let ba = before_count
                    .get(&(a2.clone(), a1.clone()))
                    .copied()
                    .unwrap_or(0);

                // Check for exclusivity (never co-occur)
                if co == 0 && trace_count > 0 {
                    exclusive_pairs.push((a1.clone(), a2.clone()));
                    continue;
                }

                // Check for sequentiality (one always before the other)
                let seq_threshold = (config.sequence_threshold * co as f64) as u64;

                if ab >= seq_threshold && ba == 0 {
                    sequential_pairs.push((a1.clone(), a2.clone()));
                } else if ba >= seq_threshold && ab == 0 {
                    sequential_pairs.push((a2.clone(), a1.clone()));
                } else if ab > 0 && ba > 0 {
                    // Both orderings observed - potential concurrency
                    let concurrent_ratio = (ab.min(ba) as f64) / (ab.max(ba) as f64);
                    if concurrent_ratio >= config.concurrency_threshold {
                        concurrent_pairs.push((a1.clone(), a2.clone()));
                    }
                }
            }
        }

        // Calculate parallelism score
        let total_pairs = activities.len() * (activities.len() - 1) / 2;
        let parallelism_score = if total_pairs > 0 {
            concurrent_pairs.len() as f64 / total_pairs as f64
        } else {
            0.0
        };

        PartialOrderResult {
            concurrent_pairs,
            sequential_pairs,
            exclusive_pairs,
            parallelism_score,
        }
    }

    /// Analyze partial orders in a single trace.
    pub fn analyze_trace(trace: &Trace) -> TracePartialOrder {
        let mut sorted_events: Vec<_> = trace.events.iter().collect();
        sorted_events.sort_by_key(|e| e.timestamp);

        let mut ordering_graph: HashMap<String, HashSet<String>> = HashMap::new();
        let mut concurrent_with: HashMap<String, HashSet<String>> = HashMap::new();

        // Build ordering relationships
        for i in 0..sorted_events.len() {
            for j in (i + 1)..sorted_events.len() {
                let a1 = &sorted_events[i].activity;
                let a2 = &sorted_events[j].activity;

                if a1 != a2 {
                    // Check if timestamps are close enough to be concurrent
                    let time_diff = sorted_events[j]
                        .timestamp
                        .saturating_sub(sorted_events[i].timestamp);

                    if time_diff == 0 {
                        // Same timestamp - concurrent
                        concurrent_with
                            .entry(a1.clone())
                            .or_default()
                            .insert(a2.clone());
                        concurrent_with
                            .entry(a2.clone())
                            .or_default()
                            .insert(a1.clone());
                    } else {
                        // Sequential
                        ordering_graph
                            .entry(a1.clone())
                            .or_default()
                            .insert(a2.clone());
                    }
                }
            }
        }

        TracePartialOrder {
            ordering_graph,
            concurrent_with,
        }
    }

    /// Detect loops in the process based on repeated activity patterns.
    pub fn detect_loops(log: &EventLog) -> Vec<LoopPattern> {
        let mut loop_patterns: HashMap<Vec<String>, u64> = HashMap::new();

        for trace in log.traces.values() {
            let mut sorted_events: Vec<_> = trace.events.iter().collect();
            sorted_events.sort_by_key(|e| e.timestamp);

            let activities: Vec<String> =
                sorted_events.iter().map(|e| e.activity.clone()).collect();

            // Find repeated subsequences
            for window_size in 2..=activities.len().min(5) {
                for start in 0..activities.len().saturating_sub(window_size * 2 - 1) {
                    let pattern: Vec<String> = activities[start..start + window_size].to_vec();

                    // Check if pattern repeats
                    let next_start = start + window_size;
                    if next_start + window_size <= activities.len() {
                        let next_pattern: Vec<String> =
                            activities[next_start..next_start + window_size].to_vec();
                        if pattern == next_pattern {
                            *loop_patterns.entry(pattern).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        loop_patterns
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .map(|(activities, count)| LoopPattern {
                activities,
                occurrence_count: count,
            })
            .collect()
    }

    /// Calculate activity independence scores.
    pub fn calculate_independence(log: &EventLog) -> HashMap<(String, String), f64> {
        let activities: Vec<String> = log
            .activities()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let mut independence_scores: HashMap<(String, String), f64> = HashMap::new();

        if activities.is_empty() || log.trace_count() == 0 {
            return independence_scores;
        }

        // Count how often activities appear in the same trace
        let mut cooccurrence_count: HashMap<(String, String), u64> = HashMap::new();
        let mut activity_count: HashMap<String, u64> = HashMap::new();

        for trace in log.traces.values() {
            let trace_activities: HashSet<_> =
                trace.events.iter().map(|e| e.activity.clone()).collect();

            for activity in &trace_activities {
                *activity_count.entry(activity.clone()).or_insert(0) += 1;
            }

            for a1 in &trace_activities {
                for a2 in &trace_activities {
                    if a1 < a2 {
                        *cooccurrence_count
                            .entry((a1.clone(), a2.clone()))
                            .or_insert(0) += 1;
                    }
                }
            }
        }

        let trace_count = log.trace_count() as f64;

        // Calculate independence using PMI-like measure
        for i in 0..activities.len() {
            for j in (i + 1)..activities.len() {
                let a1 = &activities[i];
                let a2 = &activities[j];

                let key = if a1 < a2 {
                    (a1.clone(), a2.clone())
                } else {
                    (a2.clone(), a1.clone())
                };

                let p_a1 = activity_count.get(a1).copied().unwrap_or(0) as f64 / trace_count;
                let p_a2 = activity_count.get(a2).copied().unwrap_or(0) as f64 / trace_count;
                let p_joint =
                    cooccurrence_count.get(&key).copied().unwrap_or(0) as f64 / trace_count;

                // Independence score: how different from expected co-occurrence
                let expected = p_a1 * p_a2;
                let independence = if expected > 0.0 && p_joint > 0.0 {
                    1.0 - (p_joint / expected).min(1.0)
                } else if p_joint == 0.0 {
                    1.0 // Never co-occur = independent
                } else {
                    0.0
                };

                independence_scores.insert((a1.clone(), a2.clone()), independence);
            }
        }

        independence_scores
    }
}

impl GpuKernel for PartialOrderAnalysis {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Configuration for partial order analysis.
#[derive(Debug, Clone)]
pub struct PartialOrderConfig {
    /// Threshold for considering activities concurrent (0.0-1.0).
    /// Higher values require more balanced bidirectional ordering.
    pub concurrency_threshold: f64,
    /// Threshold for considering activities sequential (0.0-1.0).
    /// Proportion of traces where ordering must be consistent.
    pub sequence_threshold: f64,
}

impl Default for PartialOrderConfig {
    fn default() -> Self {
        Self {
            concurrency_threshold: 0.5,
            sequence_threshold: 0.8,
        }
    }
}

/// Partial order information for a single trace.
#[derive(Debug, Clone)]
pub struct TracePartialOrder {
    /// Ordering graph: activity -> activities that come after it.
    pub ordering_graph: HashMap<String, HashSet<String>>,
    /// Concurrent activities: activity -> activities concurrent with it.
    pub concurrent_with: HashMap<String, HashSet<String>>,
}

/// A detected loop pattern.
#[derive(Debug, Clone)]
pub struct LoopPattern {
    /// Activities in the loop.
    pub activities: Vec<String>,
    /// Number of occurrences.
    pub occurrence_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProcessEvent;

    fn create_test_log() -> EventLog {
        let mut log = EventLog::new("test_log".to_string());

        // Trace 1: A -> B -> C -> D
        for (i, activity) in ["A", "B", "C", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: i as u64,
                case_id: "case1".to_string(),
                activity: activity.to_string(),
                timestamp: (i as u64 + 1) * 1000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Trace 2: A -> B -> C -> D (same order)
        for (i, activity) in ["A", "B", "C", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: (i + 10) as u64,
                case_id: "case2".to_string(),
                activity: activity.to_string(),
                timestamp: (i as u64 + 1) * 1000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Trace 3: A -> C -> B -> D (B and C swapped - concurrent)
        for (i, activity) in ["A", "C", "B", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: (i + 20) as u64,
                case_id: "case3".to_string(),
                activity: activity.to_string(),
                timestamp: (i as u64 + 1) * 1000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        log
    }

    fn create_exclusive_log() -> EventLog {
        let mut log = EventLog::new("exclusive_log".to_string());

        // Trace 1: A -> B -> C
        for (i, activity) in ["A", "B", "C"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: i as u64,
                case_id: "case1".to_string(),
                activity: activity.to_string(),
                timestamp: (i as u64 + 1) * 1000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Trace 2: A -> D -> E (different path)
        for (i, activity) in ["A", "D", "E"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: (i + 10) as u64,
                case_id: "case2".to_string(),
                activity: activity.to_string(),
                timestamp: (i as u64 + 1) * 1000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        log
    }

    #[test]
    fn test_partial_order_metadata() {
        let kernel = PartialOrderAnalysis::new();
        assert_eq!(kernel.metadata().id, "procint/partial-order");
        assert_eq!(kernel.metadata().domain, Domain::ProcessIntelligence);
    }

    #[test]
    fn test_concurrent_detection() {
        let log = create_test_log();
        let config = PartialOrderConfig::default();
        let result = PartialOrderAnalysis::analyze(&log, &config);

        // B and C should be detected as concurrent (appear in both orders)
        let bc_concurrent = result
            .concurrent_pairs
            .iter()
            .any(|(a, b)| (a == "B" && b == "C") || (a == "C" && b == "B"));
        assert!(bc_concurrent, "B and C should be concurrent");
    }

    #[test]
    fn test_sequential_detection() {
        let log = create_test_log();
        let config = PartialOrderConfig::default();
        let result = PartialOrderAnalysis::analyze(&log, &config);

        // A should be before B, C, D in all traces
        let a_before_d = result
            .sequential_pairs
            .iter()
            .any(|(a, b)| a == "A" && b == "D");
        assert!(a_before_d, "A should be sequential before D");
    }

    #[test]
    fn test_exclusive_detection() {
        let log = create_exclusive_log();
        let config = PartialOrderConfig::default();
        let result = PartialOrderAnalysis::analyze(&log, &config);

        // B and D should be exclusive (never in same trace)
        let bd_exclusive = result
            .exclusive_pairs
            .iter()
            .any(|(a, b)| (a == "B" && b == "D") || (a == "D" && b == "B"));
        assert!(bd_exclusive, "B and D should be exclusive");
    }

    #[test]
    fn test_parallelism_score() {
        let log = create_test_log();
        let config = PartialOrderConfig::default();
        let result = PartialOrderAnalysis::analyze(&log, &config);

        // Should have some parallelism due to B/C concurrency
        assert!(result.parallelism_score >= 0.0 && result.parallelism_score <= 1.0);
    }

    #[test]
    fn test_empty_log() {
        let log = EventLog::new("empty".to_string());
        let config = PartialOrderConfig::default();
        let result = PartialOrderAnalysis::analyze(&log, &config);

        assert!(result.concurrent_pairs.is_empty());
        assert!(result.sequential_pairs.is_empty());
        assert!(result.exclusive_pairs.is_empty());
        assert_eq!(result.parallelism_score, 0.0);
    }

    #[test]
    fn test_trace_partial_order() {
        let trace = crate::types::Trace {
            case_id: "test".to_string(),
            events: vec![
                ProcessEvent {
                    id: 1,
                    case_id: "test".to_string(),
                    activity: "A".to_string(),
                    timestamp: 1000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 2,
                    case_id: "test".to_string(),
                    activity: "B".to_string(),
                    timestamp: 2000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 3,
                    case_id: "test".to_string(),
                    activity: "C".to_string(),
                    timestamp: 3000,
                    resource: None,
                    attributes: HashMap::new(),
                },
            ],
            attributes: HashMap::new(),
        };

        let result = PartialOrderAnalysis::analyze_trace(&trace);

        // A should come before B and C
        assert!(
            result
                .ordering_graph
                .get("A")
                .map_or(false, |s| s.contains("B"))
        );
        assert!(
            result
                .ordering_graph
                .get("A")
                .map_or(false, |s| s.contains("C"))
        );
    }

    #[test]
    fn test_loop_detection() {
        let mut log = EventLog::new("loop_log".to_string());

        // Trace with loop: A -> B -> C -> B -> C -> D
        for (i, activity) in ["A", "B", "C", "B", "C", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: i as u64,
                case_id: "case1".to_string(),
                activity: activity.to_string(),
                timestamp: (i as u64 + 1) * 1000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Another trace with same loop
        for (i, activity) in ["A", "B", "C", "B", "C", "D"].iter().enumerate() {
            log.add_event(ProcessEvent {
                id: (i + 10) as u64,
                case_id: "case2".to_string(),
                activity: activity.to_string(),
                timestamp: (i as u64 + 1) * 1000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        let loops = PartialOrderAnalysis::detect_loops(&log);

        // Should detect B -> C loop
        let bc_loop = loops.iter().any(|l| l.activities == vec!["B", "C"]);
        assert!(bc_loop, "Should detect B -> C loop pattern");
    }

    #[test]
    fn test_independence_calculation() {
        let log = create_exclusive_log();
        let independence = PartialOrderAnalysis::calculate_independence(&log);

        // B and D never co-occur, should have high independence
        let bd_key = ("B".to_string(), "D".to_string());
        if let Some(&score) = independence.get(&bd_key) {
            assert_eq!(score, 1.0, "B and D should be fully independent");
        }
    }

    #[test]
    fn test_config_thresholds() {
        let log = create_test_log();

        // Strict config - fewer concurrent pairs
        let strict_config = PartialOrderConfig {
            concurrency_threshold: 0.9,
            sequence_threshold: 0.95,
        };
        let strict_result = PartialOrderAnalysis::analyze(&log, &strict_config);

        // Loose config - more concurrent pairs
        let loose_config = PartialOrderConfig {
            concurrency_threshold: 0.3,
            sequence_threshold: 0.5,
        };
        let loose_result = PartialOrderAnalysis::analyze(&log, &loose_config);

        // Loose config should detect at least as many concurrent pairs
        assert!(loose_result.concurrent_pairs.len() >= strict_result.concurrent_pairs.len());
    }
}
