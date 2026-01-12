//! Conformance checking kernel.
//!
//! This module provides conformance checking between event logs and process models:
//! - Token-based replay on DFG
//! - Petri net replay
//! - Fitness and precision calculation
//! - Deviation detection and classification

use crate::types::{
    AlignmentStep, ConformanceResult, ConformanceStats, Deviation, DeviationType,
    DirectlyFollowsGraph, EventLog, PetriNet, Trace,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Conformance Checking Kernel
// ============================================================================

/// Conformance checking kernel.
///
/// Checks how well traces in an event log conform to a process model (DFG or Petri net).
#[derive(Debug, Clone)]
pub struct ConformanceChecking {
    metadata: KernelMetadata,
}

impl Default for ConformanceChecking {
    fn default() -> Self {
        Self::new()
    }
}

impl ConformanceChecking {
    /// Create a new conformance checking kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("procint/conformance", Domain::ProcessIntelligence)
                .with_description("Multi-model conformance checking")
                .with_throughput(50_000)
                .with_latency_us(20.0),
        }
    }

    /// Check conformance of a trace against a DFG.
    pub fn check_dfg(trace: &Trace, dfg: &DirectlyFollowsGraph) -> ConformanceResult {
        let mut deviations = Vec::new();
        let mut alignment = Vec::new();
        let mut sync_moves = 0u64;
        let mut log_moves = 0u64;

        let mut events: Vec<_> = trace.events.iter().collect();
        events.sort_by_key(|e| e.timestamp);

        if events.is_empty() {
            return ConformanceResult {
                case_id: trace.case_id.clone(),
                is_conformant: true,
                fitness: 1.0,
                precision: 1.0,
                deviations,
                alignment: Some(alignment),
            };
        }

        // Check start activity
        let first_activity = &events[0].activity;
        if !dfg.start_activities.contains_key(first_activity) {
            deviations.push(Deviation {
                event_index: 0,
                activity: first_activity.clone(),
                deviation_type: DeviationType::UnexpectedActivity,
                description: format!("'{}' is not a valid start activity", first_activity),
            });
            log_moves += 1;
            alignment.push(AlignmentStep {
                log_move: Some(first_activity.clone()),
                model_move: None,
                sync: false,
                cost: 1,
            });
        } else {
            sync_moves += 1;
            alignment.push(AlignmentStep {
                log_move: Some(first_activity.clone()),
                model_move: Some(first_activity.clone()),
                sync: true,
                cost: 0,
            });
        }

        // Check directly-follows relationships
        for (i, window) in events.windows(2).enumerate() {
            let source = &window[0].activity;
            let target = &window[1].activity;

            if dfg.edge(source, target).is_some() {
                // Valid transition
                sync_moves += 1;
                alignment.push(AlignmentStep {
                    log_move: Some(target.clone()),
                    model_move: Some(target.clone()),
                    sync: true,
                    cost: 0,
                });
            } else {
                // Invalid transition
                deviations.push(Deviation {
                    event_index: i + 1,
                    activity: target.clone(),
                    deviation_type: DeviationType::WrongOrder,
                    description: format!("No edge from '{}' to '{}' in model", source, target),
                });
                log_moves += 1;
                alignment.push(AlignmentStep {
                    log_move: Some(target.clone()),
                    model_move: None,
                    sync: false,
                    cost: 1,
                });
            }
        }

        // Check end activity
        let last_activity = &events[events.len() - 1].activity;
        if !dfg.end_activities.contains_key(last_activity) {
            deviations.push(Deviation {
                event_index: events.len() - 1,
                activity: last_activity.clone(),
                deviation_type: DeviationType::UnexpectedActivity,
                description: format!("'{}' is not a valid end activity", last_activity),
            });
        }

        let total_moves = sync_moves + log_moves;
        let fitness = if total_moves > 0 {
            sync_moves as f64 / total_moves as f64
        } else {
            1.0
        };

        // Calculate precision based on how many valid options exist
        let precision = Self::calculate_dfg_precision(&events, dfg);

        ConformanceResult {
            case_id: trace.case_id.clone(),
            is_conformant: deviations.is_empty(),
            fitness,
            precision,
            deviations,
            alignment: Some(alignment),
        }
    }

    /// Check conformance of a trace against a Petri net.
    pub fn check_petri_net(trace: &Trace, net: &PetriNet) -> ConformanceResult {
        let mut deviations = Vec::new();
        let mut alignment = Vec::new();
        let mut marking = net.initial_marking.clone();

        let mut events: Vec<_> = trace.events.iter().collect();
        events.sort_by_key(|e| e.timestamp);

        let mut sync_moves = 0u64;
        let mut log_moves = 0u64;
        let mut model_moves = 0u64;

        for (i, event) in events.iter().enumerate() {
            // Find transition for this activity
            let transition = net
                .transitions
                .iter()
                .find(|t| t.label.as_ref() == Some(&event.activity));

            match transition {
                Some(t) => {
                    // Check if transition is enabled
                    let enabled = net
                        .arcs
                        .iter()
                        .filter(|a| a.target == t.id)
                        .all(|a| marking.get(&a.source).copied().unwrap_or(0) >= a.weight);

                    if enabled {
                        // Fire the transition
                        for arc in net.arcs.iter().filter(|a| a.target == t.id) {
                            if let Some(tokens) = marking.get_mut(&arc.source) {
                                *tokens = tokens.saturating_sub(arc.weight);
                            }
                        }
                        for arc in net.arcs.iter().filter(|a| a.source == t.id) {
                            *marking.entry(arc.target.clone()).or_insert(0) += arc.weight;
                        }

                        sync_moves += 1;
                        alignment.push(AlignmentStep {
                            log_move: Some(event.activity.clone()),
                            model_move: Some(t.id.clone()),
                            sync: true,
                            cost: 0,
                        });
                    } else {
                        // Transition not enabled - deviation
                        deviations.push(Deviation {
                            event_index: i,
                            activity: event.activity.clone(),
                            deviation_type: DeviationType::WrongOrder,
                            description: format!("Transition for '{}' not enabled", event.activity),
                        });
                        log_moves += 1;
                        alignment.push(AlignmentStep {
                            log_move: Some(event.activity.clone()),
                            model_move: None,
                            sync: false,
                            cost: 1,
                        });
                    }
                }
                None => {
                    // No transition for this activity
                    deviations.push(Deviation {
                        event_index: i,
                        activity: event.activity.clone(),
                        deviation_type: DeviationType::UnexpectedActivity,
                        description: format!("No transition for activity '{}'", event.activity),
                    });
                    log_moves += 1;
                    alignment.push(AlignmentStep {
                        log_move: Some(event.activity.clone()),
                        model_move: None,
                        sync: false,
                        cost: 1,
                    });
                }
            }
        }

        // Check if we reached final marking
        let reached_final = net
            .final_marking
            .iter()
            .all(|(place, &tokens)| marking.get(place).copied().unwrap_or(0) >= tokens);

        if !reached_final && !net.final_marking.is_empty() {
            // Need model moves to reach final marking
            model_moves += 1;
        }

        let total_moves = sync_moves + log_moves + model_moves;
        let fitness = if total_moves > 0 {
            sync_moves as f64 / total_moves as f64
        } else {
            1.0
        };

        let precision = if sync_moves + log_moves > 0 {
            sync_moves as f64 / (sync_moves + log_moves) as f64
        } else {
            1.0
        };

        ConformanceResult {
            case_id: trace.case_id.clone(),
            is_conformant: deviations.is_empty() && reached_final,
            fitness,
            precision,
            deviations,
            alignment: Some(alignment),
        }
    }

    /// Calculate conformance statistics for an entire log.
    pub fn check_log_dfg(log: &EventLog, dfg: &DirectlyFollowsGraph) -> ConformanceStats {
        let mut total_fitness = 0.0;
        let mut total_precision = 0.0;
        let mut conformant_count = 0u64;
        let mut deviation_counts: HashMap<DeviationType, u64> = HashMap::new();

        for trace in log.traces.values() {
            let result = Self::check_dfg(trace, dfg);

            total_fitness += result.fitness;
            total_precision += result.precision;

            if result.is_conformant {
                conformant_count += 1;
            }

            for deviation in result.deviations {
                *deviation_counts
                    .entry(deviation.deviation_type)
                    .or_insert(0) += 1;
            }
        }

        let trace_count = log.trace_count() as u64;
        let avg_fitness = if trace_count > 0 {
            total_fitness / trace_count as f64
        } else {
            0.0
        };
        let avg_precision = if trace_count > 0 {
            total_precision / trace_count as f64
        } else {
            0.0
        };

        ConformanceStats {
            trace_count,
            conformant_count,
            avg_fitness,
            avg_precision,
            deviation_counts,
        }
    }

    /// Calculate DFG precision for a trace.
    fn calculate_dfg_precision(
        events: &[&crate::types::ProcessEvent],
        dfg: &DirectlyFollowsGraph,
    ) -> f64 {
        if events.len() < 2 {
            return 1.0;
        }

        let mut total_options = 0u64;
        let mut used_options = 0u64;

        for event in events {
            let activity = &event.activity;
            let outgoing = dfg.outgoing(activity);
            if !outgoing.is_empty() {
                total_options += outgoing.len() as u64;
                used_options += 1; // Only one option used
            }
        }

        if total_options > 0 {
            used_options as f64 / total_options as f64
        } else {
            1.0
        }
    }

    /// Detect specific deviation patterns.
    pub fn classify_deviations(result: &ConformanceResult) -> DeviationSummary {
        let mut summary = DeviationSummary::default();

        for deviation in &result.deviations {
            match deviation.deviation_type {
                DeviationType::UnexpectedActivity => summary.unexpected_activities += 1,
                DeviationType::MissingActivity => summary.missing_activities += 1,
                DeviationType::WrongOrder => summary.wrong_order += 1,
                DeviationType::UnexpectedRepetition => summary.unexpected_repetitions += 1,
            }
        }

        summary.total = result.deviations.len() as u64;
        summary
    }

    /// Find the most common deviation patterns across a log.
    pub fn find_common_deviations(
        log: &EventLog,
        dfg: &DirectlyFollowsGraph,
        top_n: usize,
    ) -> Vec<CommonDeviation> {
        let mut deviation_patterns: HashMap<String, u64> = HashMap::new();

        for trace in log.traces.values() {
            let result = Self::check_dfg(trace, dfg);
            for deviation in result.deviations {
                let pattern = format!("{:?}:{}", deviation.deviation_type, deviation.activity);
                *deviation_patterns.entry(pattern).or_insert(0) += 1;
            }
        }

        let mut patterns: Vec<_> = deviation_patterns.into_iter().collect();
        patterns.sort_by(|a, b| b.1.cmp(&a.1));

        patterns
            .into_iter()
            .take(top_n)
            .map(|(pattern, count)| {
                let activity = pattern.split(':').nth(1).unwrap_or("").to_string();
                CommonDeviation {
                    pattern,
                    activity,
                    count,
                }
            })
            .collect()
    }
}

impl GpuKernel for ConformanceChecking {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Summary of deviation types.
#[derive(Debug, Clone, Default)]
pub struct DeviationSummary {
    /// Total deviations.
    pub total: u64,
    /// Unexpected activity count.
    pub unexpected_activities: u64,
    /// Missing activity count.
    pub missing_activities: u64,
    /// Wrong order count.
    pub wrong_order: u64,
    /// Unexpected repetition count.
    pub unexpected_repetitions: u64,
}

/// A common deviation pattern.
#[derive(Debug, Clone)]
pub struct CommonDeviation {
    /// Pattern description.
    pub pattern: String,
    /// Activity involved.
    pub activity: String,
    /// Occurrence count.
    pub count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dfg::DFGConstruction;
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

        // Trace 2: A -> B -> C -> D
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

        log
    }

    fn create_conformant_trace() -> Trace {
        Trace {
            case_id: "conformant".to_string(),
            events: vec![
                ProcessEvent {
                    id: 1,
                    case_id: "conformant".to_string(),
                    activity: "A".to_string(),
                    timestamp: 1000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 2,
                    case_id: "conformant".to_string(),
                    activity: "B".to_string(),
                    timestamp: 2000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 3,
                    case_id: "conformant".to_string(),
                    activity: "C".to_string(),
                    timestamp: 3000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 4,
                    case_id: "conformant".to_string(),
                    activity: "D".to_string(),
                    timestamp: 4000,
                    resource: None,
                    attributes: HashMap::new(),
                },
            ],
            attributes: HashMap::new(),
        }
    }

    fn create_non_conformant_trace() -> Trace {
        Trace {
            case_id: "non_conformant".to_string(),
            events: vec![
                ProcessEvent {
                    id: 1,
                    case_id: "non_conformant".to_string(),
                    activity: "A".to_string(),
                    timestamp: 1000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 2,
                    case_id: "non_conformant".to_string(),
                    activity: "D".to_string(), // Skips B and C
                    timestamp: 2000,
                    resource: None,
                    attributes: HashMap::new(),
                },
            ],
            attributes: HashMap::new(),
        }
    }

    #[test]
    fn test_conformance_metadata() {
        let kernel = ConformanceChecking::new();
        assert_eq!(kernel.metadata().id, "procint/conformance");
        assert_eq!(kernel.metadata().domain, Domain::ProcessIntelligence);
    }

    #[test]
    fn test_conformant_trace_dfg() {
        let log = create_test_log();
        let dfg_result = DFGConstruction::compute(&log);
        let trace = create_conformant_trace();

        let result = ConformanceChecking::check_dfg(&trace, &dfg_result.dfg);

        assert!(result.is_conformant);
        assert_eq!(result.fitness, 1.0);
        assert!(result.deviations.is_empty());
    }

    #[test]
    fn test_non_conformant_trace_dfg() {
        let log = create_test_log();
        let dfg_result = DFGConstruction::compute(&log);
        let trace = create_non_conformant_trace();

        let result = ConformanceChecking::check_dfg(&trace, &dfg_result.dfg);

        assert!(!result.is_conformant);
        assert!(result.fitness < 1.0);
        assert!(!result.deviations.is_empty());
    }

    #[test]
    fn test_fitness_calculation() {
        let log = create_test_log();
        let dfg_result = DFGConstruction::compute(&log);

        // Trace with one deviation
        let trace = Trace {
            case_id: "partial".to_string(),
            events: vec![
                ProcessEvent {
                    id: 1,
                    case_id: "partial".to_string(),
                    activity: "A".to_string(),
                    timestamp: 1000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 2,
                    case_id: "partial".to_string(),
                    activity: "B".to_string(),
                    timestamp: 2000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 3,
                    case_id: "partial".to_string(),
                    activity: "X".to_string(), // Unknown activity
                    timestamp: 3000,
                    resource: None,
                    attributes: HashMap::new(),
                },
            ],
            attributes: HashMap::new(),
        };

        let result = ConformanceChecking::check_dfg(&trace, &dfg_result.dfg);

        // Fitness should be between 0 and 1
        assert!(result.fitness > 0.0 && result.fitness < 1.0);
    }

    #[test]
    fn test_log_conformance_stats() {
        let log = create_test_log();
        let dfg_result = DFGConstruction::compute(&log);

        let stats = ConformanceChecking::check_log_dfg(&log, &dfg_result.dfg);

        assert_eq!(stats.trace_count, 2);
        assert_eq!(stats.conformant_count, 2);
        assert_eq!(stats.avg_fitness, 1.0);
    }

    #[test]
    fn test_alignment_steps() {
        let log = create_test_log();
        let dfg_result = DFGConstruction::compute(&log);
        let trace = create_conformant_trace();

        let result = ConformanceChecking::check_dfg(&trace, &dfg_result.dfg);

        let alignment = result.alignment.unwrap();
        assert_eq!(alignment.len(), 4); // A, B, C, D
        assert!(alignment.iter().all(|s| s.sync));
    }

    #[test]
    fn test_deviation_classification() {
        let log = create_test_log();
        let dfg_result = DFGConstruction::compute(&log);
        let trace = create_non_conformant_trace();

        let result = ConformanceChecking::check_dfg(&trace, &dfg_result.dfg);
        let summary = ConformanceChecking::classify_deviations(&result);

        assert!(summary.total > 0);
    }

    #[test]
    fn test_petri_net_conformance() {
        let mut net = PetriNet::new("test_net".to_string());

        // Simple sequence: p1 -> t1(A) -> p2 -> t2(B) -> p3
        net.add_place("p1".to_string(), "Start".to_string());
        net.add_place("p2".to_string(), "Middle".to_string());
        net.add_place("p3".to_string(), "End".to_string());

        net.add_transition("t1".to_string(), Some("A".to_string()));
        net.add_transition("t2".to_string(), Some("B".to_string()));

        net.add_arc("p1".to_string(), "t1".to_string(), 1);
        net.add_arc("t1".to_string(), "p2".to_string(), 1);
        net.add_arc("p2".to_string(), "t2".to_string(), 1);
        net.add_arc("t2".to_string(), "p3".to_string(), 1);

        net.initial_marking.insert("p1".to_string(), 1);
        net.final_marking.insert("p3".to_string(), 1);

        // Conformant trace
        let trace = Trace {
            case_id: "pn_test".to_string(),
            events: vec![
                ProcessEvent {
                    id: 1,
                    case_id: "pn_test".to_string(),
                    activity: "A".to_string(),
                    timestamp: 1000,
                    resource: None,
                    attributes: HashMap::new(),
                },
                ProcessEvent {
                    id: 2,
                    case_id: "pn_test".to_string(),
                    activity: "B".to_string(),
                    timestamp: 2000,
                    resource: None,
                    attributes: HashMap::new(),
                },
            ],
            attributes: HashMap::new(),
        };

        let result = ConformanceChecking::check_petri_net(&trace, &net);

        assert!(result.is_conformant);
        assert_eq!(result.fitness, 1.0);
    }

    #[test]
    fn test_empty_trace() {
        let log = create_test_log();
        let dfg_result = DFGConstruction::compute(&log);

        let trace = Trace {
            case_id: "empty".to_string(),
            events: Vec::new(),
            attributes: HashMap::new(),
        };

        let result = ConformanceChecking::check_dfg(&trace, &dfg_result.dfg);

        assert!(result.is_conformant);
        assert_eq!(result.fitness, 1.0);
    }

    #[test]
    fn test_common_deviations() {
        let mut log = EventLog::new("deviation_log".to_string());

        // Multiple traces with same deviation pattern
        for case_id in ["case1", "case2", "case3"] {
            log.add_event(ProcessEvent {
                id: 1,
                case_id: case_id.to_string(),
                activity: "A".to_string(),
                timestamp: 1000,
                resource: None,
                attributes: HashMap::new(),
            });
            log.add_event(ProcessEvent {
                id: 2,
                case_id: case_id.to_string(),
                activity: "X".to_string(), // Unknown
                timestamp: 2000,
                resource: None,
                attributes: HashMap::new(),
            });
        }

        // Build DFG from different log
        let model_log = create_test_log();
        let dfg_result = DFGConstruction::compute(&model_log);

        let common = ConformanceChecking::find_common_deviations(&log, &dfg_result.dfg, 5);

        assert!(!common.is_empty());
        assert!(common[0].count >= 3); // X appears in all 3 traces
    }
}
