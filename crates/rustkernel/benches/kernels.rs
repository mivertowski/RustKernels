//! Benchmark suite for RustKernels
//!
//! Run with: `cargo bench --package rustkernel`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;

// ============================================================================
// Graph Analytics Benchmarks
// ============================================================================

#[cfg(feature = "graph")]
mod graph_benches {
    use super::*;
    use rustkernel::graph::{
        centrality::PageRank,
        types::{GraphEdge, GraphNode, PageRankConfig},
    };
    use rustkernel::core::traits::GpuKernel;

    fn create_test_graph(node_count: usize, edge_density: f64) -> (Vec<GraphNode>, Vec<GraphEdge>) {
        let nodes: Vec<GraphNode> = (0..node_count)
            .map(|i| GraphNode {
                id: format!("n{}", i),
                attributes: HashMap::new(),
            })
            .collect();

        let edge_count = ((node_count as f64 * node_count as f64 * edge_density) as usize).max(1);
        let edges: Vec<GraphEdge> = (0..edge_count)
            .map(|i| GraphEdge {
                source: format!("n{}", i % node_count),
                target: format!("n{}", (i * 7 + 3) % node_count),
                weight: 1.0,
                attributes: HashMap::new(),
            })
            .collect();

        (nodes, edges)
    }

    pub fn pagerank_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("graph/pagerank");

        for size in [100, 500, 1000, 5000].iter() {
            let (nodes, edges) = create_test_graph(*size, 0.01);
            let config = PageRankConfig {
                damping_factor: 0.85,
                max_iterations: 20,
                convergence_threshold: 1e-6,
                personalization: None,
            };
            let kernel = PageRank::with_config(config);

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("nodes", size), size, |b, _| {
                b.iter(|| kernel.compute(black_box(&nodes), black_box(&edges)))
            });
        }

        group.finish();
    }
}

// ============================================================================
// ML Benchmarks
// ============================================================================

#[cfg(feature = "ml")]
mod ml_benches {
    use super::*;
    use rustkernel::ml::{
        clustering::KMeans,
        types::{DataPoint, KMeansConfig, KMeansInit},
    };
    use rustkernel::core::traits::GpuKernel;

    fn create_test_data(point_count: usize, dimensions: usize) -> Vec<DataPoint> {
        (0..point_count)
            .map(|i| DataPoint {
                id: format!("p{}", i),
                features: (0..dimensions)
                    .map(|d| ((i * 17 + d * 31) % 1000) as f64 / 100.0)
                    .collect(),
            })
            .collect()
    }

    pub fn kmeans_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("ml/kmeans");

        for size in [100, 500, 1000, 5000].iter() {
            let data = create_test_data(*size, 10);
            let config = KMeansConfig {
                k: 5,
                max_iterations: 50,
                convergence_threshold: 1e-4,
                initialization: KMeansInit::KMeansPlusPlus,
                seed: Some(42),
            };
            let kernel = KMeans::with_config(config);

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("points", size), size, |b, _| {
                b.iter(|| kernel.cluster(black_box(&data)))
            });
        }

        group.finish();
    }
}

// ============================================================================
// Compliance Benchmarks
// ============================================================================

#[cfg(feature = "compliance")]
mod compliance_benches {
    use super::*;
    use rustkernel::compliance::{
        aml::AMLPatternDetection,
        types::{AMLConfig, AMLPattern, Transaction, TransactionType},
    };
    use rustkernel::core::traits::GpuKernel;

    fn create_test_transactions(count: usize) -> Vec<Transaction> {
        (0..count)
            .map(|i| Transaction {
                id: format!("tx{}", i),
                from_entity: format!("entity{}", i % 100),
                to_entity: format!("entity{}", (i * 7 + 3) % 100),
                amount: ((i * 17) % 50000) as f64 + 100.0,
                currency: "USD".to_string(),
                timestamp: 1000 + i as u64 * 100,
                transaction_type: TransactionType::Transfer,
                attributes: HashMap::new(),
            })
            .collect()
    }

    pub fn aml_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("compliance/aml");

        for size in [100, 500, 1000, 5000].iter() {
            let transactions = create_test_transactions(*size);
            let config = AMLConfig {
                structuring_threshold: 10000.0,
                structuring_window: 86400,
                circular_flow_min_amount: 10000.0,
                rapid_movement_window: 3600,
                high_risk_jurisdictions: vec![],
                patterns_to_detect: vec![AMLPattern::Structuring, AMLPattern::CircularFlow],
            };
            let kernel = AMLPatternDetection::with_config(config);

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("transactions", size), size, |b, _| {
                b.iter(|| kernel.detect(black_box(&transactions)))
            });
        }

        group.finish();
    }
}

// ============================================================================
// Risk Benchmarks
// ============================================================================

#[cfg(feature = "risk")]
mod risk_benches {
    use super::*;
    use rustkernel::risk::{
        market::MonteCarloVaR,
        types::{ConfidenceLevel, Portfolio, Position, VaRConfig, VaRMethod},
    };
    use rustkernel::core::traits::GpuKernel;

    fn create_test_portfolio(position_count: usize) -> Portfolio {
        Portfolio {
            id: "test_portfolio".to_string(),
            positions: (0..position_count)
                .map(|i| Position {
                    asset_id: format!("ASSET{}", i),
                    quantity: ((i + 1) * 100) as f64,
                    current_price: 100.0 + (i as f64 * 10.0),
                    volatility: 0.15 + (i as f64 * 0.01),
                    correlation_group: Some(format!("group{}", i % 3)),
                })
                .collect(),
            base_currency: "USD".to_string(),
        }
    }

    pub fn var_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("risk/var");

        for simulations in [1000, 5000, 10000].iter() {
            let portfolio = create_test_portfolio(20);
            let config = VaRConfig {
                method: VaRMethod::MonteCarlo,
                simulations: *simulations as u32,
                time_horizon_days: 10,
                confidence_levels: vec![ConfidenceLevel::P95, ConfidenceLevel::P99],
                use_correlations: true,
                seed: Some(42),
            };
            let kernel = MonteCarloVaR::with_config(config);

            group.throughput(Throughput::Elements(*simulations as u64));
            group.bench_with_input(
                BenchmarkId::new("simulations", simulations),
                simulations,
                |b, _| b.iter(|| kernel.calculate(black_box(&portfolio))),
            );
        }

        group.finish();
    }
}

// ============================================================================
// Temporal Benchmarks
// ============================================================================

#[cfg(feature = "temporal")]
mod temporal_benches {
    use super::*;
    use rustkernel::temporal::{
        forecasting::ARIMAForecast,
        types::{ARIMAConfig, ForecastHorizon, TimeSeries},
    };
    use rustkernel::core::traits::GpuKernel;

    fn create_test_time_series(length: usize) -> TimeSeries {
        TimeSeries {
            id: "test_series".to_string(),
            timestamps: (0..length).map(|i| i as u64 * 3600).collect(),
            values: (0..length)
                .map(|i| 100.0 + (i as f64 * 0.5) + ((i as f64 * 0.1).sin() * 10.0))
                .collect(),
            frequency: Some("hourly".to_string()),
        }
    }

    pub fn arima_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("temporal/arima");

        for length in [100, 500, 1000].iter() {
            let series = create_test_time_series(*length);
            let config = ARIMAConfig {
                p: 1,
                d: 1,
                q: 1,
                seasonal: None,
                auto_select: false,
                confidence_level: 0.95,
            };
            let kernel = ARIMAForecast::with_config(config);
            let horizon = ForecastHorizon::Periods(10);

            group.throughput(Throughput::Elements(*length as u64));
            group.bench_with_input(BenchmarkId::new("series_length", length), length, |b, _| {
                b.iter(|| kernel.forecast(black_box(&series), black_box(horizon.clone())))
            });
        }

        group.finish();
    }
}

// ============================================================================
// Criterion Configuration
// ============================================================================

#[cfg(feature = "graph")]
use graph_benches::pagerank_benchmark;

#[cfg(feature = "ml")]
use ml_benches::kmeans_benchmark;

#[cfg(feature = "compliance")]
use compliance_benches::aml_benchmark;

#[cfg(feature = "risk")]
use risk_benches::var_benchmark;

#[cfg(feature = "temporal")]
use temporal_benches::arima_benchmark;

// Build criterion groups based on enabled features
#[cfg(all(
    feature = "graph",
    feature = "ml",
    feature = "compliance",
    feature = "risk",
    feature = "temporal"
))]
criterion_group!(
    benches,
    pagerank_benchmark,
    kmeans_benchmark,
    aml_benchmark,
    var_benchmark,
    arima_benchmark
);

#[cfg(all(
    feature = "graph",
    feature = "ml",
    feature = "compliance",
    feature = "risk",
    feature = "temporal"
))]
criterion_main!(benches);

// Fallback for minimal features
#[cfg(not(all(
    feature = "graph",
    feature = "ml",
    feature = "compliance",
    feature = "risk",
    feature = "temporal"
)))]
fn main() {
    println!("Enable all default features to run benchmarks:");
    println!("  cargo bench --package rustkernel --features default");
}
