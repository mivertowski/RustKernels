//! Directly-Follows Graph construction kernel.
//!
//! This module provides DFG construction from event logs:
//! - Activity frequency calculation
//! - Directly-follows relationship extraction
//! - Start/end activity identification

use crate::types::{DFGEdge, DFGResult, DirectlyFollowsGraph, EventLog, Trace};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// DFG Construction Kernel
// ============================================================================

/// DFG construction kernel.
///
/// Constructs a directly-follows graph from an event log.
#[derive(Debug, Clone)]
pub struct DFGConstruction {
    metadata: KernelMetadata,
}

impl Default for DFGConstruction {
    fn default() -> Self {
        Self::new()
    }
}

impl DFGConstruction {
    /// Create a new DFG construction kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("procint/dfg-construction", Domain::ProcessIntelligence)
                .with_description("Directly-follows graph construction")
                .with_throughput(100_000)
                .with_latency_us(50.0),
        }
    }

    /// Construct DFG from an event log.
    pub fn compute(log: &EventLog) -> DFGResult {
        let mut dfg = DirectlyFollowsGraph::new();
        let mut edge_map: HashMap<(String, String), (u64, Vec<u64>)> = HashMap::new();

        let mut event_count = 0u64;

        for trace in log.traces.values() {
            // Get sorted events
            let mut events: Vec<_> = trace.events.iter().collect();
            events.sort_by_key(|e| e.timestamp);

            event_count += events.len() as u64;

            // Track start and end activities
            if let Some(first) = events.first() {
                *dfg.start_activities
                    .entry(first.activity.clone())
                    .or_insert(0) += 1;
            }
            if let Some(last) = events.last() {
                *dfg.end_activities.entry(last.activity.clone()).or_insert(0) += 1;
            }

            // Count activity occurrences
            for event in &events {
                *dfg.activity_counts
                    .entry(event.activity.clone())
                    .or_insert(0) += 1;
            }

            // Extract directly-follows pairs
            for window in events.windows(2) {
                let source = &window[0].activity;
                let target = &window[1].activity;
                let duration = window[1].timestamp.saturating_sub(window[0].timestamp);

                let key = (source.clone(), target.clone());
                let entry = edge_map.entry(key).or_insert((0, Vec::new()));
                entry.0 += 1;
                entry.1.push(duration);
            }
        }

        // Build activities list
        dfg.activities = dfg.activity_counts.keys().cloned().collect();
        dfg.activities.sort();

        // Save edge count before consuming edge_map
        let unique_pairs = edge_map.len() as u64;

        // Build edges
        for ((source, target), (count, durations)) in edge_map {
            let avg_duration = if durations.is_empty() {
                0.0
            } else {
                durations.iter().sum::<u64>() as f64 / durations.len() as f64
            };

            dfg.edges.push(DFGEdge {
                source,
                target,
                count,
                avg_duration_ms: avg_duration,
            });
        }

        // Sort edges by count descending
        dfg.edges.sort_by(|a, b| b.count.cmp(&a.count));

        DFGResult {
            dfg,
            trace_count: log.trace_count() as u64,
            event_count,
            unique_pairs,
        }
    }

    /// Construct DFG from a single trace.
    pub fn compute_trace(trace: &Trace) -> DirectlyFollowsGraph {
        let mut dfg = DirectlyFollowsGraph::new();
        let mut edge_map: HashMap<(String, String), (u64, Vec<u64>)> = HashMap::new();

        let mut events: Vec<_> = trace.events.iter().collect();
        events.sort_by_key(|e| e.timestamp);

        // Track start and end
        if let Some(first) = events.first() {
            dfg.start_activities.insert(first.activity.clone(), 1);
        }
        if let Some(last) = events.last() {
            dfg.end_activities.insert(last.activity.clone(), 1);
        }

        // Count activities
        for event in &events {
            *dfg.activity_counts
                .entry(event.activity.clone())
                .or_insert(0) += 1;
        }

        // Extract pairs
        for window in events.windows(2) {
            let source = &window[0].activity;
            let target = &window[1].activity;
            let duration = window[1].timestamp.saturating_sub(window[0].timestamp);

            let key = (source.clone(), target.clone());
            let entry = edge_map.entry(key).or_insert((0, Vec::new()));
            entry.0 += 1;
            entry.1.push(duration);
        }

        // Build activities and edges
        dfg.activities = dfg.activity_counts.keys().cloned().collect();
        dfg.activities.sort();

        for ((source, target), (count, durations)) in edge_map {
            let avg_duration = if durations.is_empty() {
                0.0
            } else {
                durations.iter().sum::<u64>() as f64 / durations.len() as f64
            };

            dfg.edges.push(DFGEdge {
                source,
                target,
                count,
                avg_duration_ms: avg_duration,
            });
        }

        dfg
    }

    /// Filter DFG by minimum edge frequency.
    pub fn filter_by_frequency(dfg: &DirectlyFollowsGraph, min_count: u64) -> DirectlyFollowsGraph {
        let mut filtered = DirectlyFollowsGraph::new();

        // Keep activities that appear in filtered edges
        let mut active_activities = std::collections::HashSet::new();

        filtered.edges = dfg
            .edges
            .iter()
            .filter(|e| e.count >= min_count)
            .map(|e| {
                active_activities.insert(e.source.clone());
                active_activities.insert(e.target.clone());
                e.clone()
            })
            .collect();

        filtered.activities = active_activities.into_iter().collect();
        filtered.activities.sort();

        // Filter activity counts
        filtered.activity_counts = dfg
            .activity_counts
            .iter()
            .filter(|(k, _)| filtered.activities.contains(*k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        // Filter start/end activities
        filtered.start_activities = dfg
            .start_activities
            .iter()
            .filter(|(k, _)| filtered.activities.contains(*k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        filtered.end_activities = dfg
            .end_activities
            .iter()
            .filter(|(k, _)| filtered.activities.contains(*k))
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        filtered
    }

    /// Calculate graph metrics.
    pub fn calculate_metrics(dfg: &DirectlyFollowsGraph) -> DFGMetrics {
        let node_count = dfg.activities.len();
        let edge_count = dfg.edges.len();

        let max_possible_edges = node_count * node_count;
        let density = if max_possible_edges > 0 {
            edge_count as f64 / max_possible_edges as f64
        } else {
            0.0
        };

        let total_edge_weight: u64 = dfg.edges.iter().map(|e| e.count).sum();
        let avg_edge_weight = if edge_count > 0 {
            total_edge_weight as f64 / edge_count as f64
        } else {
            0.0
        };

        DFGMetrics {
            node_count,
            edge_count,
            density,
            avg_edge_weight,
            start_activity_count: dfg.start_activities.len(),
            end_activity_count: dfg.end_activities.len(),
        }
    }
}

impl GpuKernel for DFGConstruction {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// DFG metrics.
#[derive(Debug, Clone)]
pub struct DFGMetrics {
    /// Number of nodes (activities).
    pub node_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Graph density.
    pub density: f64,
    /// Average edge weight.
    pub avg_edge_weight: f64,
    /// Number of start activities.
    pub start_activity_count: usize,
    /// Number of end activities.
    pub end_activity_count: usize,
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

        // Trace 2: A -> B -> C -> D (same pattern)
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

        // Trace 3: A -> B -> E -> D (different middle)
        for (i, activity) in ["A", "B", "E", "D"].iter().enumerate() {
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

    #[test]
    fn test_dfg_construction_metadata() {
        let kernel = DFGConstruction::new();
        assert_eq!(kernel.metadata().id, "procint/dfg-construction");
        assert_eq!(kernel.metadata().domain, Domain::ProcessIntelligence);
    }

    #[test]
    fn test_dfg_construction() {
        let log = create_test_log();
        let result = DFGConstruction::compute(&log);

        assert_eq!(result.trace_count, 3);
        assert_eq!(result.event_count, 12);

        // Should have activities A, B, C, D, E
        assert_eq!(result.dfg.activities.len(), 5);
    }

    #[test]
    fn test_dfg_edges() {
        let log = create_test_log();
        let result = DFGConstruction::compute(&log);

        // A -> B should appear in all 3 traces
        let ab_edge = result.dfg.edge("A", "B");
        assert!(ab_edge.is_some());
        assert_eq!(ab_edge.unwrap().count, 3);

        // B -> C should appear in 2 traces
        let bc_edge = result.dfg.edge("B", "C");
        assert!(bc_edge.is_some());
        assert_eq!(bc_edge.unwrap().count, 2);

        // B -> E should appear in 1 trace
        let be_edge = result.dfg.edge("B", "E");
        assert!(be_edge.is_some());
        assert_eq!(be_edge.unwrap().count, 1);
    }

    #[test]
    fn test_start_end_activities() {
        let log = create_test_log();
        let result = DFGConstruction::compute(&log);

        // A is start activity in all traces
        assert_eq!(result.dfg.start_activities.get("A").copied(), Some(3));

        // D is end activity in all traces
        assert_eq!(result.dfg.end_activities.get("D").copied(), Some(3));
    }

    #[test]
    fn test_activity_counts() {
        let log = create_test_log();
        let result = DFGConstruction::compute(&log);

        // A appears 3 times (once per trace)
        assert_eq!(result.dfg.activity_counts.get("A").copied(), Some(3));

        // C appears 2 times
        assert_eq!(result.dfg.activity_counts.get("C").copied(), Some(2));

        // E appears 1 time
        assert_eq!(result.dfg.activity_counts.get("E").copied(), Some(1));
    }

    #[test]
    fn test_filter_by_frequency() {
        let log = create_test_log();
        let result = DFGConstruction::compute(&log);

        // Filter to only edges with count >= 2
        let filtered = DFGConstruction::filter_by_frequency(&result.dfg, 2);

        // B -> E should be removed (count=1)
        assert!(filtered.edge("B", "E").is_none());

        // A -> B should remain (count=3)
        assert!(filtered.edge("A", "B").is_some());
    }

    #[test]
    fn test_dfg_metrics() {
        let log = create_test_log();
        let result = DFGConstruction::compute(&log);
        let metrics = DFGConstruction::calculate_metrics(&result.dfg);

        assert_eq!(metrics.node_count, 5);
        assert!(metrics.edge_count > 0);
        assert!(metrics.density > 0.0 && metrics.density <= 1.0);
        assert_eq!(metrics.start_activity_count, 1); // Only A is start
        assert_eq!(metrics.end_activity_count, 1); // Only D is end
    }

    #[test]
    fn test_single_trace() {
        let trace = Trace {
            case_id: "test".to_string(),
            events: vec![
                ProcessEvent {
                    id: 1,
                    case_id: "test".to_string(),
                    activity: "X".to_string(),
                    timestamp: 1000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 2,
                    case_id: "test".to_string(),
                    activity: "Y".to_string(),
                    timestamp: 2000,
                    resource: None,
                    attributes: HashMap::new(),
                },
            ],
            attributes: HashMap::new(),
        };

        let dfg = DFGConstruction::compute_trace(&trace);

        assert_eq!(dfg.activities.len(), 2);
        assert!(dfg.edge("X", "Y").is_some());
        assert_eq!(dfg.start_activities.get("X").copied(), Some(1));
        assert_eq!(dfg.end_activities.get("Y").copied(), Some(1));
    }

    #[test]
    fn test_empty_log() {
        let log = EventLog::new("empty".to_string());
        let result = DFGConstruction::compute(&log);

        assert_eq!(result.trace_count, 0);
        assert_eq!(result.event_count, 0);
        assert!(result.dfg.activities.is_empty());
    }
}
