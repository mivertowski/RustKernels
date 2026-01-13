//! Benchmark suite for RustKernels
//!
//! Run with: `cargo bench --package rustkernel`
//!
//! TODO: Update benchmarks to match current kernel APIs.
//! The benchmark code was written for an earlier API version and needs to be
//! updated to match the current struct definitions and function signatures.

use criterion::{Criterion, criterion_group, criterion_main};

/// Placeholder benchmark - actual benchmarks are disabled pending API updates.
fn placeholder_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // Placeholder - real benchmarks will be added when APIs stabilize
            std::hint::black_box(1 + 1)
        })
    });
}

criterion_group!(benches, placeholder_benchmark);
criterion_main!(benches);
