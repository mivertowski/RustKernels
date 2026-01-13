//! Digital Twin Simulation Kernel
//!
//! GPU-accelerated process simulation for what-if analysis,
//! resource optimization, and predictive process management.
//!
//! # Features
//! - Monte Carlo simulation of process execution
//! - Resource contention modeling
//! - What-if scenario analysis
//! - Bottleneck prediction
//! - Queue dynamics simulation

use rustkernel_core::error::Result;
use rustkernel_core::traits::GpuKernel;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata};
use std::collections::HashMap;

/// Digital Twin process simulation kernel.
///
/// Simulates process execution using discrete event simulation
/// with Monte Carlo sampling for activity durations.
#[derive(Debug)]
pub struct DigitalTwin {
    metadata: KernelMetadata,
}

impl DigitalTwin {
    /// Create a new DigitalTwin kernel.
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("procint/digital-twin", Domain::ProcessIntelligence)
                .with_description("Process simulation for what-if analysis and optimization"),
        }
    }

    /// Run process simulation.
    pub fn simulate(
        &self,
        model: &ProcessModel,
        config: &SimulationConfig,
    ) -> Result<SimulationResult> {
        // Run multiple simulation replications
        let mut all_traces: Vec<SimulatedTrace> = Vec::new();
        let mut resource_utilizations: HashMap<String, f64> = HashMap::new();
        let mut activity_stats: HashMap<String, ActivityStats> = HashMap::new();
        let mut bottleneck_scores: HashMap<String, f64> = HashMap::new();

        // Initialize resource utilization tracking
        for resource in &model.resources {
            resource_utilizations.insert(resource.id.clone(), 0.0);
        }

        // Initialize activity stats
        for activity in &model.activities {
            activity_stats.insert(
                activity.id.clone(),
                ActivityStats {
                    count: 0,
                    total_duration: 0.0,
                    total_waiting: 0.0,
                    min_duration: f64::MAX,
                    max_duration: 0.0,
                },
            );
        }

        // Run replications (GPU would parallelize this)
        for replication in 0..config.replications {
            let seed = config.seed.unwrap_or(42) + replication as u64;
            let traces = self.run_replication(model, config, seed)?;

            // Collect statistics from traces
            for trace in &traces {
                for event in &trace.events {
                    // Update activity stats
                    if let Some(stats) = activity_stats.get_mut(&event.activity_id) {
                        stats.count += 1;
                        stats.total_duration += event.duration;
                        stats.total_waiting += event.waiting_time;
                        stats.min_duration = stats.min_duration.min(event.duration);
                        stats.max_duration = stats.max_duration.max(event.duration);
                    }

                    // Update resource utilization
                    if let Some(resource_id) = &event.resource_id {
                        if let Some(util) = resource_utilizations.get_mut(resource_id) {
                            *util += event.duration;
                        }
                    }
                }
            }

            all_traces.extend(traces);
        }

        // Normalize statistics
        let total_time = config.simulation_duration * config.replications as f64;

        for util in resource_utilizations.values_mut() {
            *util /= total_time;
        }

        // Compute bottleneck scores based on waiting time ratio
        for (activity_id, stats) in &activity_stats {
            if stats.count > 0 {
                let avg_waiting = stats.total_waiting / stats.count as f64;
                let avg_duration = stats.total_duration / stats.count as f64;
                let bottleneck_score = if avg_duration > 0.0 {
                    avg_waiting / (avg_waiting + avg_duration)
                } else {
                    0.0
                };
                bottleneck_scores.insert(activity_id.clone(), bottleneck_score);
            }
        }

        // Find primary bottleneck
        let primary_bottleneck = bottleneck_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone());

        // Compute KPIs
        let kpis = self.compute_kpis(&all_traces, config);

        Ok(SimulationResult {
            traces: all_traces,
            kpis,
            resource_utilization: resource_utilizations,
            bottleneck_scores,
            primary_bottleneck,
            replications_completed: config.replications,
        })
    }

    fn run_replication(
        &self,
        model: &ProcessModel,
        config: &SimulationConfig,
        seed: u64,
    ) -> Result<Vec<SimulatedTrace>> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut traces = Vec::new();
        let mut current_time = 0.0;
        let mut case_id = 0;

        // Resource availability tracking (time when each resource becomes free)
        let mut resource_free_at: HashMap<String, f64> = HashMap::new();
        for resource in &model.resources {
            for _ in 0..resource.capacity {
                let key = format!("{}_{}", resource.id, resource_free_at.len());
                resource_free_at.insert(key, 0.0);
            }
        }

        // Generate arrivals and process cases
        while current_time < config.simulation_duration {
            // Inter-arrival time (exponential distribution)
            let inter_arrival = self.sample_exponential(&mut rng, config.arrival_rate);
            current_time += inter_arrival;

            if current_time >= config.simulation_duration {
                break;
            }

            // Process this case through the model
            let trace = self.process_case(
                model,
                case_id,
                current_time,
                &mut resource_free_at,
                &mut rng,
            );
            traces.push(trace);
            case_id += 1;
        }

        Ok(traces)
    }

    fn process_case(
        &self,
        model: &ProcessModel,
        case_id: u32,
        arrival_time: f64,
        resource_free_at: &mut HashMap<String, f64>,
        rng: &mut rand::rngs::StdRng,
    ) -> SimulatedTrace {
        let mut events = Vec::new();
        let mut current_time = arrival_time;

        // Simple sequential execution through activities
        // (Real implementation would follow process model structure)
        for activity in &model.activities {
            // Sample activity duration
            let duration = self.sample_duration(&activity.duration_dist, rng);

            // Find earliest available resource
            let (resource_key, waiting_time) =
                self.find_resource(activity, current_time, resource_free_at);

            let start_time = current_time + waiting_time;
            let end_time = start_time + duration;

            // Update resource availability
            if let Some(key) = &resource_key {
                resource_free_at.insert(key.clone(), end_time);
            }

            events.push(SimulatedEvent {
                activity_id: activity.id.clone(),
                start_time,
                end_time,
                duration,
                waiting_time,
                resource_id: resource_key.map(|k| k.split('_').next().unwrap_or(&k).to_string()),
            });

            current_time = end_time;
        }

        SimulatedTrace {
            case_id,
            arrival_time,
            completion_time: current_time,
            cycle_time: current_time - arrival_time,
            events,
        }
    }

    fn find_resource(
        &self,
        activity: &Activity,
        current_time: f64,
        resource_free_at: &HashMap<String, f64>,
    ) -> (Option<String>, f64) {
        if activity.required_resources.is_empty() {
            return (None, 0.0);
        }

        // Find earliest available resource of required type
        let mut best_key: Option<String> = None;
        let mut best_time = f64::MAX;

        for required_resource in &activity.required_resources {
            for (key, free_time) in resource_free_at {
                if key.starts_with(required_resource) && *free_time < best_time {
                    best_time = *free_time;
                    best_key = Some(key.clone());
                }
            }
        }

        if let Some(key) = &best_key {
            let waiting = (best_time - current_time).max(0.0);
            (Some(key.clone()), waiting)
        } else {
            (None, 0.0)
        }
    }

    fn sample_exponential(&self, rng: &mut rand::rngs::StdRng, rate: f64) -> f64 {
        use rand::Rng;
        let u: f64 = rng.random();
        -u.ln() / rate
    }

    fn sample_duration(&self, dist: &DurationDistribution, rng: &mut rand::rngs::StdRng) -> f64 {
        use rand::Rng;
        match dist {
            DurationDistribution::Constant(value) => *value,
            DurationDistribution::Uniform { min, max } => rng.random_range(*min..=*max),
            DurationDistribution::Normal { mean, std_dev } => {
                // Box-Muller transform
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (mean + std_dev * z).max(0.0)
            }
            DurationDistribution::Exponential { mean } => {
                let u: f64 = rng.random();
                -mean * u.ln()
            }
            DurationDistribution::Triangular { min, mode, max } => {
                let u: f64 = rng.random();
                let fc = (mode - min) / (max - min);
                if u < fc {
                    min + ((max - min) * (mode - min) * u).sqrt()
                } else {
                    max - ((max - min) * (max - mode) * (1.0 - u)).sqrt()
                }
            }
        }
    }

    fn compute_kpis(&self, traces: &[SimulatedTrace], config: &SimulationConfig) -> ProcessKPIs {
        if traces.is_empty() {
            return ProcessKPIs::default();
        }

        let cycle_times: Vec<f64> = traces.iter().map(|t| t.cycle_time).collect();
        let n = cycle_times.len() as f64;

        let avg_cycle_time = cycle_times.iter().sum::<f64>() / n;
        let min_cycle_time = cycle_times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_cycle_time = cycle_times
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Standard deviation
        let variance = cycle_times
            .iter()
            .map(|t| (t - avg_cycle_time).powi(2))
            .sum::<f64>()
            / n;
        let std_cycle_time = variance.sqrt();

        // Percentiles
        let mut sorted = cycle_times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50 = self.percentile(&sorted, 0.5);
        let p90 = self.percentile(&sorted, 0.9);
        let p95 = self.percentile(&sorted, 0.95);
        let p99 = self.percentile(&sorted, 0.99);

        // Throughput
        let throughput = traces.len() as f64 / config.simulation_duration;

        // WIP (Work in Progress) - Little's Law approximation
        let wip = throughput * avg_cycle_time;

        // Service level (percentage completing within target)
        let service_level = if let Some(target) = config.target_cycle_time {
            let within_target = cycle_times.iter().filter(|&&t| t <= target).count();
            within_target as f64 / n
        } else {
            1.0
        };

        ProcessKPIs {
            avg_cycle_time,
            min_cycle_time,
            max_cycle_time,
            std_cycle_time,
            p50_cycle_time: p50,
            p90_cycle_time: p90,
            p95_cycle_time: p95,
            p99_cycle_time: p99,
            throughput,
            work_in_progress: wip,
            service_level,
            total_cases: traces.len(),
        }
    }

    fn percentile(&self, sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((sorted.len() - 1) as f64 * p) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

impl Default for DigitalTwin {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for DigitalTwin {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }

    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Process model definition.
#[derive(Debug, Clone)]
pub struct ProcessModel {
    /// Model identifier.
    pub id: String,
    /// Activities in the process.
    pub activities: Vec<Activity>,
    /// Available resources.
    pub resources: Vec<Resource>,
    /// ProcessTransitions between activities.
    pub transitions: Vec<ProcessTransition>,
}

/// Activity definition.
#[derive(Debug, Clone)]
pub struct Activity {
    /// Activity identifier.
    pub id: String,
    /// Activity name.
    pub name: String,
    /// Duration distribution.
    pub duration_dist: DurationDistribution,
    /// Required resource types.
    pub required_resources: Vec<String>,
}

/// Resource definition.
#[derive(Debug, Clone)]
pub struct Resource {
    /// Resource identifier.
    pub id: String,
    /// Resource name.
    pub name: String,
    /// Number of instances.
    pub capacity: u32,
    /// Cost per time unit.
    pub cost_per_unit: f64,
}

/// ProcessTransition between activities.
#[derive(Debug, Clone)]
pub struct ProcessTransition {
    /// Source activity.
    pub from: String,
    /// Target activity.
    pub to: String,
    /// Probability (for XOR splits).
    pub probability: f64,
}

/// Duration distribution types.
#[derive(Debug, Clone)]
pub enum DurationDistribution {
    /// Fixed duration.
    Constant(f64),
    /// Uniform distribution between min and max.
    Uniform {
        /// Minimum duration.
        min: f64,
        /// Maximum duration.
        max: f64,
    },
    /// Normal (Gaussian) distribution.
    Normal {
        /// Mean of the distribution.
        mean: f64,
        /// Standard deviation.
        std_dev: f64,
    },
    /// Exponential distribution.
    Exponential {
        /// Mean of the distribution.
        mean: f64,
    },
    /// Triangular distribution.
    Triangular {
        /// Minimum duration.
        min: f64,
        /// Most likely duration.
        mode: f64,
        /// Maximum duration.
        max: f64,
    },
}

/// Simulation configuration.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of simulation replications.
    pub replications: u32,
    /// Duration of each replication (time units).
    pub simulation_duration: f64,
    /// Arrival rate (cases per time unit).
    pub arrival_rate: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Target cycle time for service level calculation.
    pub target_cycle_time: Option<f64>,
    /// Warm-up period to exclude from statistics.
    pub warmup_period: Option<f64>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            replications: 100,
            simulation_duration: 480.0, // 8 hours
            arrival_rate: 0.1,          // 10 cases per hour
            seed: Some(42),
            target_cycle_time: None,
            warmup_period: None,
        }
    }
}

/// Simulation result.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Simulated traces.
    pub traces: Vec<SimulatedTrace>,
    /// Process KPIs.
    pub kpis: ProcessKPIs,
    /// Resource utilization (0-1).
    pub resource_utilization: HashMap<String, f64>,
    /// Bottleneck scores by activity.
    pub bottleneck_scores: HashMap<String, f64>,
    /// Primary bottleneck activity.
    pub primary_bottleneck: Option<String>,
    /// Number of replications completed.
    pub replications_completed: u32,
}

/// Simulated process trace.
#[derive(Debug, Clone)]
pub struct SimulatedTrace {
    /// Case identifier.
    pub case_id: u32,
    /// Arrival time.
    pub arrival_time: f64,
    /// Completion time.
    pub completion_time: f64,
    /// Total cycle time.
    pub cycle_time: f64,
    /// Simulated events.
    pub events: Vec<SimulatedEvent>,
}

/// Simulated event.
#[derive(Debug, Clone)]
pub struct SimulatedEvent {
    /// Activity identifier.
    pub activity_id: String,
    /// Start time.
    pub start_time: f64,
    /// End time.
    pub end_time: f64,
    /// Actual duration.
    pub duration: f64,
    /// Time spent waiting for resources.
    pub waiting_time: f64,
    /// Resource used (if any).
    pub resource_id: Option<String>,
}

/// Process key performance indicators.
#[derive(Debug, Clone, Default)]
pub struct ProcessKPIs {
    /// Average cycle time.
    pub avg_cycle_time: f64,
    /// Minimum cycle time.
    pub min_cycle_time: f64,
    /// Maximum cycle time.
    pub max_cycle_time: f64,
    /// Standard deviation of cycle time.
    pub std_cycle_time: f64,
    /// 50th percentile cycle time.
    pub p50_cycle_time: f64,
    /// 90th percentile cycle time.
    pub p90_cycle_time: f64,
    /// 95th percentile cycle time.
    pub p95_cycle_time: f64,
    /// 99th percentile cycle time.
    pub p99_cycle_time: f64,
    /// Throughput (cases per time unit).
    pub throughput: f64,
    /// Average work in progress.
    pub work_in_progress: f64,
    /// Service level (fraction within target).
    pub service_level: f64,
    /// Total cases processed.
    pub total_cases: usize,
}

/// Internal activity statistics tracking.
struct ActivityStats {
    count: usize,
    total_duration: f64,
    total_waiting: f64,
    min_duration: f64,
    max_duration: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> ProcessModel {
        ProcessModel {
            id: "test_process".to_string(),
            activities: vec![
                Activity {
                    id: "receive".to_string(),
                    name: "Receive Request".to_string(),
                    duration_dist: DurationDistribution::Normal {
                        mean: 5.0,
                        std_dev: 1.0,
                    },
                    required_resources: vec!["clerk".to_string()],
                },
                Activity {
                    id: "review".to_string(),
                    name: "Review Request".to_string(),
                    duration_dist: DurationDistribution::Triangular {
                        min: 10.0,
                        mode: 20.0,
                        max: 45.0,
                    },
                    required_resources: vec!["analyst".to_string()],
                },
                Activity {
                    id: "approve".to_string(),
                    name: "Approve Request".to_string(),
                    duration_dist: DurationDistribution::Exponential { mean: 8.0 },
                    required_resources: vec!["manager".to_string()],
                },
            ],
            resources: vec![
                Resource {
                    id: "clerk".to_string(),
                    name: "Clerk".to_string(),
                    capacity: 3,
                    cost_per_unit: 25.0,
                },
                Resource {
                    id: "analyst".to_string(),
                    name: "Analyst".to_string(),
                    capacity: 2,
                    cost_per_unit: 50.0,
                },
                Resource {
                    id: "manager".to_string(),
                    name: "Manager".to_string(),
                    capacity: 1,
                    cost_per_unit: 100.0,
                },
            ],
            transitions: vec![
                ProcessTransition {
                    from: "receive".to_string(),
                    to: "review".to_string(),
                    probability: 1.0,
                },
                ProcessTransition {
                    from: "review".to_string(),
                    to: "approve".to_string(),
                    probability: 1.0,
                },
            ],
        }
    }

    #[test]
    fn test_digital_twin_metadata() {
        let kernel = DigitalTwin::new();
        let metadata = kernel.metadata();
        assert_eq!(metadata.id, "procint/digital-twin");
        assert_eq!(metadata.domain, Domain::ProcessIntelligence);
    }

    #[test]
    fn test_simulation_basic() {
        let kernel = DigitalTwin::new();
        let model = create_test_model();
        let config = SimulationConfig {
            replications: 10,
            simulation_duration: 100.0,
            arrival_rate: 0.2,
            seed: Some(42),
            ..Default::default()
        };

        let result = kernel.simulate(&model, &config).unwrap();

        assert!(!result.traces.is_empty());
        assert!(result.kpis.avg_cycle_time > 0.0);
        assert!(result.kpis.throughput > 0.0);
        assert_eq!(result.replications_completed, 10);
    }

    #[test]
    fn test_resource_utilization() {
        let kernel = DigitalTwin::new();
        let model = create_test_model();
        let config = SimulationConfig {
            replications: 50,
            simulation_duration: 200.0,
            arrival_rate: 0.15,
            seed: Some(123),
            ..Default::default()
        };

        let result = kernel.simulate(&model, &config).unwrap();

        // Check that resource utilization is computed
        assert!(!result.resource_utilization.is_empty());

        // Utilization should be between 0 and some reasonable upper bound
        for (_resource, util) in &result.resource_utilization {
            assert!(*util >= 0.0);
        }
    }

    #[test]
    fn test_bottleneck_detection() {
        let kernel = DigitalTwin::new();
        let model = create_test_model();
        let config = SimulationConfig {
            replications: 20,
            simulation_duration: 150.0,
            arrival_rate: 0.25, // High arrival rate to create congestion
            seed: Some(456),
            ..Default::default()
        };

        let result = kernel.simulate(&model, &config).unwrap();

        // Should identify some bottleneck
        assert!(!result.bottleneck_scores.is_empty());

        // Bottleneck scores should be between 0 and 1
        for (_activity, score) in &result.bottleneck_scores {
            assert!(*score >= 0.0 && *score <= 1.0);
        }
    }

    #[test]
    fn test_kpis_calculation() {
        let kernel = DigitalTwin::new();
        let model = create_test_model();
        let config = SimulationConfig {
            replications: 30,
            simulation_duration: 300.0,
            arrival_rate: 0.1,
            seed: Some(789),
            target_cycle_time: Some(100.0),
            ..Default::default()
        };

        let result = kernel.simulate(&model, &config).unwrap();

        // Verify KPIs are reasonable
        assert!(result.kpis.min_cycle_time <= result.kpis.avg_cycle_time);
        assert!(result.kpis.avg_cycle_time <= result.kpis.max_cycle_time);
        assert!(result.kpis.p50_cycle_time <= result.kpis.p90_cycle_time);
        assert!(result.kpis.p90_cycle_time <= result.kpis.p95_cycle_time);
        assert!(result.kpis.service_level >= 0.0 && result.kpis.service_level <= 1.0);
    }

    #[test]
    fn test_duration_distributions() {
        let kernel = DigitalTwin::new();
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Test constant
        let constant = DurationDistribution::Constant(10.0);
        assert_eq!(kernel.sample_duration(&constant, &mut rng), 10.0);

        // Test uniform
        let uniform = DurationDistribution::Uniform {
            min: 5.0,
            max: 15.0,
        };
        let sample = kernel.sample_duration(&uniform, &mut rng);
        assert!(sample >= 5.0 && sample <= 15.0);

        // Test normal
        let normal = DurationDistribution::Normal {
            mean: 10.0,
            std_dev: 2.0,
        };
        let sample = kernel.sample_duration(&normal, &mut rng);
        assert!(sample >= 0.0); // Should be non-negative due to max(0.0)

        // Test exponential
        let exponential = DurationDistribution::Exponential { mean: 10.0 };
        let sample = kernel.sample_duration(&exponential, &mut rng);
        assert!(sample >= 0.0);

        // Test triangular
        let triangular = DurationDistribution::Triangular {
            min: 5.0,
            mode: 10.0,
            max: 20.0,
        };
        let sample = kernel.sample_duration(&triangular, &mut rng);
        assert!(sample >= 5.0 && sample <= 20.0);
    }
}
