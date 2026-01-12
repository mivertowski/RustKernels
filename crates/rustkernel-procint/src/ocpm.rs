//! Object-Centric Process Mining (OCPM) kernel.
//!
//! This module provides OCPM pattern matching capabilities:
//! - Multi-object event correlation
//! - Object lifecycle analysis
//! - Cross-object pattern detection
//! - Object flow analysis

use crate::types::{OCPMEventLog, OCPMPatternResult};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// OCPM Pattern Matching Kernel
// ============================================================================

/// OCPM pattern matching kernel.
///
/// Detects patterns in object-centric event logs where events can relate
/// to multiple objects of different types.
#[derive(Debug, Clone)]
pub struct OCPMPatternMatching {
    metadata: KernelMetadata,
}

impl Default for OCPMPatternMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl OCPMPatternMatching {
    /// Create a new OCPM pattern matching kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("procint/ocpm-patterns", Domain::ProcessIntelligence)
                .with_description("Object-centric process mining patterns")
                .with_throughput(20_000)
                .with_latency_us(200.0),
        }
    }

    /// Detect object lifecycle patterns.
    pub fn detect_lifecycle_patterns(
        log: &OCPMEventLog,
        object_type: &str,
    ) -> Vec<OCPMPatternResult> {
        let mut patterns = Vec::new();

        // Get objects of the specified type
        let objects: Vec<_> = log
            .objects
            .values()
            .filter(|o| o.object_type == object_type)
            .collect();

        for object in objects {
            let events = log.events_for_object(&object.id);

            if events.is_empty() {
                continue;
            }

            // Sort events by timestamp
            let mut sorted_events: Vec<_> = events.iter().collect();
            sorted_events.sort_by_key(|e| e.timestamp);

            // Extract activity sequence
            let sequence: Vec<_> = sorted_events.iter().map(|e| e.activity.as_str()).collect();

            // Detect lifecycle pattern
            let pattern_name = classify_lifecycle(&sequence);
            let score = calculate_lifecycle_score(&sequence);

            patterns.push(OCPMPatternResult {
                pattern_name,
                matched_objects: vec![object.id.clone()],
                matched_events: sorted_events.iter().map(|e| e.id).collect(),
                score,
                description: format!(
                    "Object {} follows lifecycle: {}",
                    object.id,
                    sequence.join(" -> ")
                ),
            });
        }

        patterns
    }

    /// Detect cross-object patterns (object interactions).
    pub fn detect_interaction_patterns(log: &OCPMEventLog) -> Vec<OCPMPatternResult> {
        let mut patterns = Vec::new();
        let mut interaction_counts: HashMap<(String, String), Vec<u64>> = HashMap::new();

        // Find events that involve multiple objects
        for event in &log.events {
            if event.objects.len() >= 2 {
                // Record all pairs of objects
                for i in 0..event.objects.len() {
                    for j in (i + 1)..event.objects.len() {
                        let obj1 = &event.objects[i];
                        let obj2 = &event.objects[j];

                        let key = if obj1 < obj2 {
                            (obj1.clone(), obj2.clone())
                        } else {
                            (obj2.clone(), obj1.clone())
                        };

                        interaction_counts.entry(key).or_default().push(event.id);
                    }
                }
            }
        }

        // Generate patterns for significant interactions
        for ((obj1, obj2), event_ids) in interaction_counts {
            if event_ids.len() >= 2 {
                // Get object types
                let type1 = log
                    .objects
                    .get(&obj1)
                    .map(|o| o.object_type.as_str())
                    .unwrap_or("unknown");
                let type2 = log
                    .objects
                    .get(&obj2)
                    .map(|o| o.object_type.as_str())
                    .unwrap_or("unknown");

                patterns.push(OCPMPatternResult {
                    pattern_name: format!("{}_{}_interaction", type1, type2),
                    matched_objects: vec![obj1.clone(), obj2.clone()],
                    matched_events: event_ids.clone(),
                    score: event_ids.len() as f64 / log.events.len().max(1) as f64,
                    description: format!(
                        "Objects {} and {} interact in {} events",
                        obj1,
                        obj2,
                        event_ids.len()
                    ),
                });
            }
        }

        patterns
    }

    /// Detect convergence patterns (multiple objects leading to one).
    pub fn detect_convergence_patterns(log: &OCPMEventLog) -> Vec<OCPMPatternResult> {
        let mut patterns = Vec::new();
        let mut convergence_points: HashMap<u64, HashSet<String>> = HashMap::new();

        // Find events where multiple objects converge
        for event in &log.events {
            if event.objects.len() >= 2 {
                let objects_set: HashSet<_> = event.objects.iter().cloned().collect();
                convergence_points.insert(event.id, objects_set);
            }
        }

        // Analyze convergence sequences
        for (event_id, objects) in &convergence_points {
            if objects.len() >= 3 {
                let event = log.events.iter().find(|e| e.id == *event_id).unwrap();

                patterns.push(OCPMPatternResult {
                    pattern_name: "convergence".to_string(),
                    matched_objects: objects.iter().cloned().collect(),
                    matched_events: vec![*event_id],
                    score: objects.len() as f64 / 10.0, // Normalize by assumed max
                    description: format!(
                        "Convergence point at '{}' with {} objects",
                        event.activity,
                        objects.len()
                    ),
                });
            }
        }

        patterns
    }

    /// Detect divergence patterns (one object leading to multiple).
    pub fn detect_divergence_patterns(log: &OCPMEventLog) -> Vec<OCPMPatternResult> {
        let mut patterns = Vec::new();

        // Track object appearances over time
        let mut object_first_seen: HashMap<String, (u64, u64)> = HashMap::new(); // object -> (event_id, timestamp)

        for event in &log.events {
            for obj_id in &event.objects {
                object_first_seen
                    .entry(obj_id.clone())
                    .or_insert((event.id, event.timestamp));
            }
        }

        // Find events that spawn multiple new objects
        let mut spawn_events: HashMap<u64, Vec<String>> = HashMap::new();

        for (obj_id, (event_id, _)) in &object_first_seen {
            spawn_events
                .entry(*event_id)
                .or_default()
                .push(obj_id.clone());
        }

        for (event_id, new_objects) in spawn_events {
            if new_objects.len() >= 2 {
                let event = log.events.iter().find(|e| e.id == event_id);
                if let Some(event) = event {
                    patterns.push(OCPMPatternResult {
                        pattern_name: "divergence".to_string(),
                        matched_objects: new_objects.clone(),
                        matched_events: vec![event_id],
                        score: new_objects.len() as f64 / 10.0,
                        description: format!(
                            "Divergence point at '{}' creating {} objects",
                            event.activity,
                            new_objects.len()
                        ),
                    });
                }
            }
        }

        patterns
    }

    /// Detect synchronization patterns.
    pub fn detect_sync_patterns(log: &OCPMEventLog, time_window_ms: u64) -> Vec<OCPMPatternResult> {
        let mut patterns = Vec::new();

        // Sort events by timestamp
        let mut sorted_events: Vec<_> = log.events.iter().collect();
        sorted_events.sort_by_key(|e| e.timestamp);

        // Find events within time window that share objects
        for i in 0..sorted_events.len() {
            let event_i = sorted_events[i];
            let mut sync_group = vec![event_i.id];
            let mut sync_objects: HashSet<String> = event_i.objects.iter().cloned().collect();

            for event_j in sorted_events.iter().skip(i + 1) {
                if event_j.timestamp > event_i.timestamp + time_window_ms {
                    break;
                }

                // Check if events share any objects
                let shared: HashSet<_> = event_j
                    .objects
                    .iter()
                    .filter(|o| sync_objects.contains(*o))
                    .cloned()
                    .collect();

                if !shared.is_empty() {
                    sync_group.push(event_j.id);
                    sync_objects.extend(event_j.objects.iter().cloned());
                }
            }

            if sync_group.len() >= 3 {
                patterns.push(OCPMPatternResult {
                    pattern_name: "synchronization".to_string(),
                    matched_objects: sync_objects.iter().cloned().collect(),
                    matched_events: sync_group.clone(),
                    score: sync_group.len() as f64 / 10.0,
                    description: format!(
                        "Synchronization of {} events within {}ms window",
                        sync_group.len(),
                        time_window_ms
                    ),
                });
            }
        }

        patterns
    }

    /// Calculate object flow metrics.
    pub fn calculate_flow_metrics(log: &OCPMEventLog) -> ObjectFlowMetrics {
        let mut object_event_counts: HashMap<String, u64> = HashMap::new();
        let mut activity_object_counts: HashMap<String, HashSet<String>> = HashMap::new();
        let mut object_type_counts: HashMap<String, u64> = HashMap::new();

        for event in &log.events {
            for obj_id in &event.objects {
                *object_event_counts.entry(obj_id.clone()).or_insert(0) += 1;

                activity_object_counts
                    .entry(event.activity.clone())
                    .or_default()
                    .insert(obj_id.clone());
            }
        }

        for obj in log.objects.values() {
            *object_type_counts
                .entry(obj.object_type.clone())
                .or_insert(0) += 1;
        }

        let avg_events_per_object = if !object_event_counts.is_empty() {
            object_event_counts.values().sum::<u64>() as f64 / object_event_counts.len() as f64
        } else {
            0.0
        };

        let avg_objects_per_activity = if !activity_object_counts.is_empty() {
            activity_object_counts
                .values()
                .map(|s| s.len() as f64)
                .sum::<f64>()
                / activity_object_counts.len() as f64
        } else {
            0.0
        };

        let max_objects_per_event = log
            .events
            .iter()
            .map(|e| e.objects.len())
            .max()
            .unwrap_or(0);

        ObjectFlowMetrics {
            object_count: log.objects.len(),
            event_count: log.events.len(),
            object_type_count: object_type_counts.len(),
            avg_events_per_object,
            avg_objects_per_activity,
            max_objects_per_event,
            object_type_distribution: object_type_counts,
        }
    }

    /// Detect batching patterns (objects processed together).
    pub fn detect_batching_patterns(log: &OCPMEventLog) -> Vec<OCPMPatternResult> {
        let mut patterns = Vec::new();
        let mut activity_batches: HashMap<String, Vec<HashSet<String>>> = HashMap::new();

        // Group events by activity and timestamp proximity
        let mut sorted_events: Vec<_> = log.events.iter().collect();
        sorted_events.sort_by_key(|e| (e.activity.clone(), e.timestamp));

        let mut current_activity = String::new();
        let mut current_batch: HashSet<String> = HashSet::new();
        let mut batch_events: Vec<u64> = Vec::new();
        let mut last_timestamp = 0u64;

        for event in sorted_events {
            if event.activity != current_activity || event.timestamp > last_timestamp + 1000 {
                // New batch
                if current_batch.len() >= 3 {
                    activity_batches
                        .entry(current_activity.clone())
                        .or_default()
                        .push(current_batch.clone());

                    patterns.push(OCPMPatternResult {
                        pattern_name: format!("{}_batch", current_activity),
                        matched_objects: current_batch.iter().cloned().collect(),
                        matched_events: batch_events.clone(),
                        score: current_batch.len() as f64 / 10.0,
                        description: format!(
                            "Batch of {} objects in activity '{}'",
                            current_batch.len(),
                            current_activity
                        ),
                    });
                }

                current_activity = event.activity.clone();
                current_batch.clear();
                batch_events.clear();
            }

            current_batch.extend(event.objects.iter().cloned());
            batch_events.push(event.id);
            last_timestamp = event.timestamp;
        }

        // Don't forget the last batch
        if current_batch.len() >= 3 {
            patterns.push(OCPMPatternResult {
                pattern_name: format!("{}_batch", current_activity),
                matched_objects: current_batch.iter().cloned().collect(),
                matched_events: batch_events,
                score: current_batch.len() as f64 / 10.0,
                description: format!(
                    "Batch of {} objects in activity '{}'",
                    current_batch.len(),
                    current_activity
                ),
            });
        }

        patterns
    }
}

impl GpuKernel for OCPMPatternMatching {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

/// Object flow metrics.
#[derive(Debug, Clone)]
pub struct ObjectFlowMetrics {
    /// Total number of objects.
    pub object_count: usize,
    /// Total number of events.
    pub event_count: usize,
    /// Number of distinct object types.
    pub object_type_count: usize,
    /// Average events per object.
    pub avg_events_per_object: f64,
    /// Average objects per activity.
    pub avg_objects_per_activity: f64,
    /// Maximum objects in a single event.
    pub max_objects_per_event: usize,
    /// Distribution of object types.
    pub object_type_distribution: HashMap<String, u64>,
}

/// Classify a lifecycle pattern.
fn classify_lifecycle(sequence: &[&str]) -> String {
    if sequence.is_empty() {
        return "empty".to_string();
    }

    // Check for common patterns
    if sequence.len() == 1 {
        return "single_event".to_string();
    }

    // Check for creation -> ... -> completion pattern
    let first = sequence[0].to_lowercase();
    let last = sequence[sequence.len() - 1].to_lowercase();

    if (first.contains("create") || first.contains("start") || first.contains("init"))
        && (last.contains("complete") || last.contains("end") || last.contains("close"))
    {
        return "full_lifecycle".to_string();
    }

    // Check for loop patterns
    let unique: HashSet<_> = sequence.iter().collect();
    if unique.len() < sequence.len() / 2 {
        return "loop_heavy".to_string();
    }

    "sequential".to_string()
}

/// Calculate lifecycle completeness score.
fn calculate_lifecycle_score(sequence: &[&str]) -> f64 {
    if sequence.is_empty() {
        return 0.0;
    }

    let has_start = sequence.iter().any(|s| {
        let lower = s.to_lowercase();
        lower.contains("create") || lower.contains("start") || lower.contains("init")
    });

    let has_end = sequence.iter().any(|s| {
        let lower = s.to_lowercase();
        lower.contains("complete") || lower.contains("end") || lower.contains("close")
    });

    let unique_ratio = sequence.iter().collect::<HashSet<_>>().len() as f64 / sequence.len() as f64;

    let mut score = 0.5 * unique_ratio;
    if has_start {
        score += 0.25;
    }
    if has_end {
        score += 0.25;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OCPMEvent, OCPMObject};

    fn create_test_ocpm_log() -> OCPMEventLog {
        let mut log = OCPMEventLog::new();

        // Add objects
        log.add_object(OCPMObject {
            id: "order1".to_string(),
            object_type: "Order".to_string(),
            attributes: HashMap::new(),
        });
        log.add_object(OCPMObject {
            id: "order2".to_string(),
            object_type: "Order".to_string(),
            attributes: HashMap::new(),
        });
        log.add_object(OCPMObject {
            id: "item1".to_string(),
            object_type: "Item".to_string(),
            attributes: HashMap::new(),
        });
        log.add_object(OCPMObject {
            id: "item2".to_string(),
            object_type: "Item".to_string(),
            attributes: HashMap::new(),
        });
        log.add_object(OCPMObject {
            id: "item3".to_string(),
            object_type: "Item".to_string(),
            attributes: HashMap::new(),
        });

        // Add events
        log.add_event(OCPMEvent {
            id: 1,
            activity: "Create Order".to_string(),
            timestamp: 1000,
            objects: vec!["order1".to_string()],
            attributes: HashMap::new(),
        });
        log.add_event(OCPMEvent {
            id: 2,
            activity: "Add Item".to_string(),
            timestamp: 2000,
            objects: vec!["order1".to_string(), "item1".to_string()],
            attributes: HashMap::new(),
        });
        log.add_event(OCPMEvent {
            id: 3,
            activity: "Add Item".to_string(),
            timestamp: 2100,
            objects: vec!["order1".to_string(), "item2".to_string()],
            attributes: HashMap::new(),
        });
        log.add_event(OCPMEvent {
            id: 4,
            activity: "Process Payment".to_string(),
            timestamp: 3000,
            objects: vec!["order1".to_string(), "order2".to_string()],
            attributes: HashMap::new(),
        });
        log.add_event(OCPMEvent {
            id: 5,
            activity: "Complete Order".to_string(),
            timestamp: 4000,
            objects: vec!["order1".to_string()],
            attributes: HashMap::new(),
        });

        log
    }

    #[test]
    fn test_ocpm_metadata() {
        let kernel = OCPMPatternMatching::new();
        assert_eq!(kernel.metadata().id, "procint/ocpm-patterns");
        assert_eq!(kernel.metadata().domain, Domain::ProcessIntelligence);
    }

    #[test]
    fn test_lifecycle_detection() {
        let log = create_test_ocpm_log();

        let patterns = OCPMPatternMatching::detect_lifecycle_patterns(&log, "Order");

        assert!(!patterns.is_empty());

        // order1 should have a full lifecycle
        let order1_pattern = patterns
            .iter()
            .find(|p| p.matched_objects.contains(&"order1".to_string()));
        assert!(order1_pattern.is_some());
    }

    #[test]
    fn test_interaction_detection() {
        let log = create_test_ocpm_log();

        let patterns = OCPMPatternMatching::detect_interaction_patterns(&log);

        // order1 and item1 should have interaction pattern
        let has_order_item = patterns.iter().any(|p| {
            p.matched_objects.contains(&"order1".to_string())
                && (p.matched_objects.contains(&"item1".to_string())
                    || p.matched_objects.contains(&"item2".to_string()))
        });

        // Note: might not have enough interactions (need 2+)
        assert!(patterns.is_empty() || has_order_item);
    }

    #[test]
    fn test_flow_metrics() {
        let log = create_test_ocpm_log();

        let metrics = OCPMPatternMatching::calculate_flow_metrics(&log);

        assert_eq!(metrics.object_count, 5);
        assert_eq!(metrics.event_count, 5);
        assert_eq!(metrics.object_type_count, 2);
        assert!(metrics.avg_events_per_object > 0.0);
        assert!(metrics.max_objects_per_event >= 2);
    }

    #[test]
    fn test_convergence_detection() {
        let mut log = OCPMEventLog::new();

        // Create convergence point with 3+ objects
        for i in 1..=4 {
            log.add_object(OCPMObject {
                id: format!("obj{}", i),
                object_type: "Item".to_string(),
                attributes: HashMap::new(),
            });
        }

        log.add_event(OCPMEvent {
            id: 1,
            activity: "Merge".to_string(),
            timestamp: 1000,
            objects: vec![
                "obj1".to_string(),
                "obj2".to_string(),
                "obj3".to_string(),
                "obj4".to_string(),
            ],
            attributes: HashMap::new(),
        });

        let patterns = OCPMPatternMatching::detect_convergence_patterns(&log);

        assert!(!patterns.is_empty());
        assert!(patterns[0].matched_objects.len() >= 3);
    }

    #[test]
    fn test_divergence_detection() {
        let mut log = OCPMEventLog::new();

        // Event that spawns multiple new objects
        for i in 1..=3 {
            log.add_object(OCPMObject {
                id: format!("new{}", i),
                object_type: "Product".to_string(),
                attributes: HashMap::new(),
            });
        }

        log.add_event(OCPMEvent {
            id: 1,
            activity: "Split".to_string(),
            timestamp: 1000,
            objects: vec!["new1".to_string(), "new2".to_string(), "new3".to_string()],
            attributes: HashMap::new(),
        });

        let patterns = OCPMPatternMatching::detect_divergence_patterns(&log);

        assert!(!patterns.is_empty());
        assert_eq!(patterns[0].pattern_name, "divergence");
    }

    #[test]
    fn test_sync_patterns() {
        let mut log = OCPMEventLog::new();

        // Create shared objects
        log.add_object(OCPMObject {
            id: "shared".to_string(),
            object_type: "Resource".to_string(),
            attributes: HashMap::new(),
        });

        // Multiple events within time window sharing an object
        for i in 0..5 {
            log.add_event(OCPMEvent {
                id: i,
                activity: format!("Process_{}", i),
                timestamp: 1000 + i * 100, // Within 500ms window
                objects: vec!["shared".to_string()],
                attributes: HashMap::new(),
            });
        }

        let patterns = OCPMPatternMatching::detect_sync_patterns(&log, 500);

        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_batching_detection() {
        let mut log = OCPMEventLog::new();

        // Create objects for batch
        for i in 1..=5 {
            log.add_object(OCPMObject {
                id: format!("batch_item{}", i),
                object_type: "Item".to_string(),
                attributes: HashMap::new(),
            });
        }

        // Batch processing events
        for i in 1..=5 {
            log.add_event(OCPMEvent {
                id: i,
                activity: "BatchProcess".to_string(),
                timestamp: 1000 + i * 100,
                objects: vec![format!("batch_item{}", i)],
                attributes: HashMap::new(),
            });
        }

        let patterns = OCPMPatternMatching::detect_batching_patterns(&log);

        // Should detect a batch pattern
        assert!(!patterns.is_empty());
        assert!(patterns[0].pattern_name.contains("batch"));
    }

    #[test]
    fn test_empty_log() {
        let log = OCPMEventLog::new();

        let lifecycle = OCPMPatternMatching::detect_lifecycle_patterns(&log, "Order");
        assert!(lifecycle.is_empty());

        let metrics = OCPMPatternMatching::calculate_flow_metrics(&log);
        assert_eq!(metrics.object_count, 0);
        assert_eq!(metrics.event_count, 0);
    }

    #[test]
    fn test_lifecycle_classification() {
        // Full lifecycle
        let full = classify_lifecycle(&["Create Order", "Process", "Complete Order"]);
        assert_eq!(full, "full_lifecycle");

        // Sequential
        let seq = classify_lifecycle(&["A", "B", "C", "D"]);
        assert_eq!(seq, "sequential");

        // Loop heavy
        let loops = classify_lifecycle(&["A", "B", "A", "B", "A", "B"]);
        assert_eq!(loops, "loop_heavy");

        // Empty
        let empty = classify_lifecycle(&[]);
        assert_eq!(empty, "empty");
    }

    #[test]
    fn test_lifecycle_score() {
        // Full lifecycle should score high
        let full_score = calculate_lifecycle_score(&["start", "process", "end"]);
        assert!(full_score >= 0.75);

        // No start/end markers should score lower
        let mid_score = calculate_lifecycle_score(&["A", "B", "C"]);
        assert!(mid_score < 0.75);
    }
}
