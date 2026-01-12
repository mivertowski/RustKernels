//! Process intelligence types and data structures.

use std::collections::{HashMap, HashSet};

// ============================================================================
// Event Log Types
// ============================================================================

/// A process event in a trace.
#[derive(Debug, Clone)]
pub struct ProcessEvent {
    /// Event ID.
    pub id: u64,
    /// Case/trace ID.
    pub case_id: String,
    /// Activity name.
    pub activity: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Resource (who performed the activity).
    pub resource: Option<String>,
    /// Additional attributes.
    pub attributes: HashMap<String, String>,
}

/// A trace (sequence of events for a case).
#[derive(Debug, Clone)]
pub struct Trace {
    /// Case ID.
    pub case_id: String,
    /// Events in order.
    pub events: Vec<ProcessEvent>,
    /// Trace attributes.
    pub attributes: HashMap<String, String>,
}

impl Trace {
    /// Create a new trace.
    pub fn new(case_id: String) -> Self {
        Self {
            case_id,
            events: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Add an event to the trace.
    pub fn add_event(&mut self, event: ProcessEvent) {
        self.events.push(event);
    }

    /// Get activity sequence.
    pub fn activity_sequence(&self) -> Vec<&str> {
        self.events.iter().map(|e| e.activity.as_str()).collect()
    }

    /// Sort events by timestamp.
    pub fn sort_by_timestamp(&mut self) {
        self.events.sort_by_key(|e| e.timestamp);
    }
}

/// An event log containing multiple traces.
#[derive(Debug, Clone)]
pub struct EventLog {
    /// Log name.
    pub name: String,
    /// Traces by case ID.
    pub traces: HashMap<String, Trace>,
    /// Log attributes.
    pub attributes: HashMap<String, String>,
}

impl EventLog {
    /// Create a new event log.
    pub fn new(name: String) -> Self {
        Self {
            name,
            traces: HashMap::new(),
            attributes: HashMap::new(),
        }
    }

    /// Add an event to the log.
    pub fn add_event(&mut self, event: ProcessEvent) {
        let trace = self
            .traces
            .entry(event.case_id.clone())
            .or_insert_with(|| Trace::new(event.case_id.clone()));
        trace.add_event(event);
    }

    /// Get all unique activities.
    pub fn activities(&self) -> HashSet<&str> {
        self.traces
            .values()
            .flat_map(|t| t.events.iter().map(|e| e.activity.as_str()))
            .collect()
    }

    /// Get trace count.
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// Get event count.
    pub fn event_count(&self) -> usize {
        self.traces.values().map(|t| t.events.len()).sum()
    }
}

// ============================================================================
// Directly-Follows Graph Types
// ============================================================================

/// A directly-follows graph (DFG).
#[derive(Debug, Clone)]
pub struct DirectlyFollowsGraph {
    /// Activities (nodes).
    pub activities: Vec<String>,
    /// Edges (from_activity, to_activity, count).
    pub edges: Vec<DFGEdge>,
    /// Start activities with frequency.
    pub start_activities: HashMap<String, u64>,
    /// End activities with frequency.
    pub end_activities: HashMap<String, u64>,
    /// Activity frequencies.
    pub activity_counts: HashMap<String, u64>,
}

/// An edge in the DFG.
#[derive(Debug, Clone)]
pub struct DFGEdge {
    /// Source activity.
    pub source: String,
    /// Target activity.
    pub target: String,
    /// Frequency count.
    pub count: u64,
    /// Average time between activities (ms).
    pub avg_duration_ms: f64,
}

impl DirectlyFollowsGraph {
    /// Create a new DFG.
    pub fn new() -> Self {
        Self {
            activities: Vec::new(),
            edges: Vec::new(),
            start_activities: HashMap::new(),
            end_activities: HashMap::new(),
            activity_counts: HashMap::new(),
        }
    }

    /// Get outgoing edges from an activity.
    pub fn outgoing(&self, activity: &str) -> Vec<&DFGEdge> {
        self.edges.iter().filter(|e| e.source == activity).collect()
    }

    /// Get incoming edges to an activity.
    pub fn incoming(&self, activity: &str) -> Vec<&DFGEdge> {
        self.edges.iter().filter(|e| e.target == activity).collect()
    }

    /// Get edge between two activities.
    pub fn edge(&self, source: &str, target: &str) -> Option<&DFGEdge> {
        self.edges
            .iter()
            .find(|e| e.source == source && e.target == target)
    }
}

impl Default for DirectlyFollowsGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// DFG construction result with statistics.
#[derive(Debug, Clone)]
pub struct DFGResult {
    /// The constructed DFG.
    pub dfg: DirectlyFollowsGraph,
    /// Number of traces processed.
    pub trace_count: u64,
    /// Number of events processed.
    pub event_count: u64,
    /// Number of unique activity pairs.
    pub unique_pairs: u64,
}

// ============================================================================
// Process Model Types
// ============================================================================

/// A Petri net place.
#[derive(Debug, Clone)]
pub struct Place {
    /// Place ID.
    pub id: String,
    /// Place name.
    pub name: String,
    /// Current token count.
    pub tokens: u32,
}

/// A Petri net transition.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Transition ID.
    pub id: String,
    /// Activity label (None for silent transitions).
    pub label: Option<String>,
    /// Is this transition visible?
    pub visible: bool,
}

/// An arc in a Petri net.
#[derive(Debug, Clone)]
pub struct Arc {
    /// Source ID (place or transition).
    pub source: String,
    /// Target ID (place or transition).
    pub target: String,
    /// Arc weight.
    pub weight: u32,
}

/// A Petri net process model.
#[derive(Debug, Clone)]
pub struct PetriNet {
    /// Model name.
    pub name: String,
    /// Places.
    pub places: Vec<Place>,
    /// Transitions.
    pub transitions: Vec<Transition>,
    /// Arcs.
    pub arcs: Vec<Arc>,
    /// Initial marking (place_id -> tokens).
    pub initial_marking: HashMap<String, u32>,
    /// Final marking (place_id -> tokens).
    pub final_marking: HashMap<String, u32>,
}

impl PetriNet {
    /// Create a new Petri net.
    pub fn new(name: String) -> Self {
        Self {
            name,
            places: Vec::new(),
            transitions: Vec::new(),
            arcs: Vec::new(),
            initial_marking: HashMap::new(),
            final_marking: HashMap::new(),
        }
    }

    /// Add a place.
    pub fn add_place(&mut self, id: String, name: String) {
        self.places.push(Place {
            id,
            name,
            tokens: 0,
        });
    }

    /// Add a transition.
    pub fn add_transition(&mut self, id: String, label: Option<String>) {
        self.transitions.push(Transition {
            id,
            label: label.clone(),
            visible: label.is_some(),
        });
    }

    /// Add an arc.
    pub fn add_arc(&mut self, source: String, target: String, weight: u32) {
        self.arcs.push(Arc {
            source,
            target,
            weight,
        });
    }

    /// Get enabled transitions for current marking.
    pub fn enabled_transitions(&self, marking: &HashMap<String, u32>) -> Vec<&Transition> {
        self.transitions
            .iter()
            .filter(|t| {
                // Check if all input places have enough tokens
                self.arcs
                    .iter()
                    .filter(|a| a.target == t.id)
                    .all(|a| marking.get(&a.source).copied().unwrap_or(0) >= a.weight)
            })
            .collect()
    }
}

// ============================================================================
// Conformance Types
// ============================================================================

/// Conformance checking result.
#[derive(Debug, Clone)]
pub struct ConformanceResult {
    /// Trace ID.
    pub case_id: String,
    /// Is the trace conformant?
    pub is_conformant: bool,
    /// Fitness score (0-1).
    pub fitness: f64,
    /// Precision score (0-1).
    pub precision: f64,
    /// Deviations found.
    pub deviations: Vec<Deviation>,
    /// Alignment (if computed).
    pub alignment: Option<Vec<AlignmentStep>>,
}

/// A deviation from the model.
#[derive(Debug, Clone)]
pub struct Deviation {
    /// Event index in trace.
    pub event_index: usize,
    /// Activity that deviated.
    pub activity: String,
    /// Type of deviation.
    pub deviation_type: DeviationType,
    /// Description.
    pub description: String,
}

/// Type of deviation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviationType {
    /// Activity not allowed by model.
    UnexpectedActivity,
    /// Expected activity was missing.
    MissingActivity,
    /// Wrong order of activities.
    WrongOrder,
    /// Activity should not repeat.
    UnexpectedRepetition,
}

/// A step in an alignment.
#[derive(Debug, Clone)]
pub struct AlignmentStep {
    /// Log move (activity from trace).
    pub log_move: Option<String>,
    /// Model move (transition from model).
    pub model_move: Option<String>,
    /// Is this a synchronous move?
    pub sync: bool,
    /// Cost of this move.
    pub cost: u32,
}

/// Aggregate conformance statistics.
#[derive(Debug, Clone)]
pub struct ConformanceStats {
    /// Number of traces checked.
    pub trace_count: u64,
    /// Number of conformant traces.
    pub conformant_count: u64,
    /// Average fitness.
    pub avg_fitness: f64,
    /// Average precision.
    pub avg_precision: f64,
    /// Deviation breakdown by type.
    pub deviation_counts: HashMap<DeviationType, u64>,
}

// ============================================================================
// Partial Order Types
// ============================================================================

/// Partial order analysis result.
#[derive(Debug, Clone)]
pub struct PartialOrderResult {
    /// Concurrent activity pairs.
    pub concurrent_pairs: Vec<(String, String)>,
    /// Sequential activity pairs (A before B).
    pub sequential_pairs: Vec<(String, String)>,
    /// Exclusive activity pairs (never in same trace).
    pub exclusive_pairs: Vec<(String, String)>,
    /// Parallelism score (0-1).
    pub parallelism_score: f64,
}

// ============================================================================
// Object-Centric Process Mining Types
// ============================================================================

/// An object in OCPM.
#[derive(Debug, Clone)]
pub struct OCPMObject {
    /// Object ID.
    pub id: String,
    /// Object type.
    pub object_type: String,
    /// Object attributes.
    pub attributes: HashMap<String, String>,
}

/// An OCPM event (can relate to multiple objects).
#[derive(Debug, Clone)]
pub struct OCPMEvent {
    /// Event ID.
    pub id: u64,
    /// Activity name.
    pub activity: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Related object IDs.
    pub objects: Vec<String>,
    /// Attributes.
    pub attributes: HashMap<String, String>,
}

/// Object-centric event log.
#[derive(Debug, Clone)]
pub struct OCPMEventLog {
    /// Events.
    pub events: Vec<OCPMEvent>,
    /// Objects.
    pub objects: HashMap<String, OCPMObject>,
    /// Object types.
    pub object_types: HashSet<String>,
}

impl OCPMEventLog {
    /// Create a new OCPM event log.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            objects: HashMap::new(),
            object_types: HashSet::new(),
        }
    }

    /// Add an object.
    pub fn add_object(&mut self, object: OCPMObject) {
        self.object_types.insert(object.object_type.clone());
        self.objects.insert(object.id.clone(), object);
    }

    /// Add an event.
    pub fn add_event(&mut self, event: OCPMEvent) {
        self.events.push(event);
    }

    /// Get events for an object.
    pub fn events_for_object(&self, object_id: &str) -> Vec<&OCPMEvent> {
        self.events
            .iter()
            .filter(|e| e.objects.contains(&object_id.to_string()))
            .collect()
    }
}

impl Default for OCPMEventLog {
    fn default() -> Self {
        Self::new()
    }
}

/// OCPM pattern match result.
#[derive(Debug, Clone)]
pub struct OCPMPatternResult {
    /// Pattern name.
    pub pattern_name: String,
    /// Matched object IDs.
    pub matched_objects: Vec<String>,
    /// Matched event IDs.
    pub matched_events: Vec<u64>,
    /// Pattern score.
    pub score: f64,
    /// Description.
    pub description: String,
}
