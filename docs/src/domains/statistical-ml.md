# Statistical ML

**Crate**: `rustkernel-ml`
**Kernels**: 8
**Feature**: `ml` (included in default features)

Machine learning kernels for clustering, anomaly detection, and regression.

## Kernel Overview

### Clustering (3)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| KMeans | `ml/kmeans` | Batch, Ring | K-means++ clustering |
| DBSCAN | `ml/dbscan` | Batch | Density-based clustering |
| HierarchicalClustering | `ml/hierarchical-clustering` | Batch | Agglomerative clustering |

### Anomaly Detection (3)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| IsolationForest | `ml/isolation-forest` | Batch | Tree-based anomaly detection |
| LocalOutlierFactor | `ml/local-outlier-factor` | Batch, Ring | Density-based outlier detection |
| EnsembleVoting | `ml/ensemble-voting` | Batch, Ring | Combine multiple detectors |

### Regression (2)

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| LinearRegression | `ml/linear-regression` | Batch, Ring | Ordinary least squares |
| RidgeRegression | `ml/ridge-regression` | Batch, Ring | L2-regularized regression |

---

## Kernel Details

### KMeans

Partitions data into K clusters using the K-means++ initialization.

**ID**: `ml/kmeans`
**Modes**: Batch, Ring
**Throughput**: ~500,000 points/sec

#### Input

```rust
pub struct KMeansInput {
    /// Data points as flattened array
    pub points: Vec<f64>,
    /// Number of dimensions per point
    pub dimensions: u32,
    /// Number of clusters
    pub k: u32,
    /// Maximum iterations
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
}
```

#### Output

```rust
pub struct KMeansOutput {
    /// Cluster assignment per point
    pub assignments: Vec<u32>,
    /// Centroids (k * dimensions)
    pub centroids: Vec<f64>,
    /// Iterations performed
    pub iterations: u32,
    /// Inertia (sum of squared distances)
    pub inertia: f64,
}
```

#### Example

```rust
use rustkernel::ml::clustering::{KMeans, KMeansInput};

let kernel = KMeans::new();

let input = KMeansInput {
    points: vec![
        1.0, 2.0,  // Point 0
        1.5, 1.8,  // Point 1
        5.0, 8.0,  // Point 2
        6.0, 9.0,  // Point 3
    ],
    dimensions: 2,
    k: 2,
    max_iterations: 100,
    tolerance: 1e-4,
};

let result = kernel.execute(input).await?;
println!("Clusters: {:?}", result.assignments);
```

---

### IsolationForest

Detects anomalies by isolating observations using random forests.

**ID**: `ml/isolation-forest`
**Modes**: Batch

#### Input

```rust
pub struct IsolationForestInput {
    pub points: Vec<f64>,
    pub dimensions: u32,
    /// Number of trees
    pub n_estimators: u32,
    /// Subsample size
    pub max_samples: u32,
    /// Contamination ratio (expected anomaly rate)
    pub contamination: f64,
}
```

#### Output

```rust
pub struct IsolationForestOutput {
    /// Anomaly scores (higher = more anomalous)
    pub scores: Vec<f64>,
    /// Binary labels (-1 = anomaly, 1 = normal)
    pub labels: Vec<i32>,
}
```

---

### LocalOutlierFactor

Measures local density deviation to identify outliers.

**ID**: `ml/local-outlier-factor`
**Modes**: Batch, Ring

#### Example

```rust
use rustkernel::ml::anomaly::{LocalOutlierFactor, LOFInput};

let kernel = LocalOutlierFactor::new();

let result = kernel.execute(LOFInput {
    points: data_points,
    dimensions: 3,
    k_neighbors: 20,
}).await?;

// Scores > 1.0 indicate outliers
let outliers: Vec<usize> = result.scores
    .iter()
    .enumerate()
    .filter(|(_, &s)| s > 1.5)
    .map(|(i, _)| i)
    .collect();
```

---

### LinearRegression

Fits a linear model using ordinary least squares.

**ID**: `ml/linear-regression`
**Modes**: Batch, Ring

#### Input

```rust
pub struct LinearRegressionInput {
    /// Feature matrix (n_samples * n_features)
    pub features: Vec<f64>,
    /// Target values (n_samples)
    pub targets: Vec<f64>,
    /// Number of features
    pub n_features: u32,
    /// Whether to fit intercept
    pub fit_intercept: bool,
}
```

#### Output

```rust
pub struct LinearRegressionOutput {
    /// Coefficients (n_features, or n_features + 1 with intercept)
    pub coefficients: Vec<f64>,
    /// Intercept (if fit_intercept = true)
    pub intercept: f64,
    /// R-squared score
    pub r_squared: f64,
}
```

---

## Ring Mode for Streaming ML

Ring mode enables online learning scenarios:

```rust
use rustkernel::ml::clustering::KMeansRing;

let ring = KMeansRing::new(k: 5, dimensions: 3);

// Stream data points
for point in incoming_stream {
    // Assign to nearest cluster (sub-millisecond)
    let cluster = ring.assign_point(point).await?;

    // Periodically update centroids
    if should_update_centroids() {
        ring.update_centroids().await?;
    }
}
```

## Performance Considerations

1. **Dimensionality**: High dimensions slow down distance calculations
2. **Memory**: KMeans stores all points; for very large datasets, consider mini-batch
3. **Initialization**: K-means++ is more expensive but gives better results
4. **GPU utilization**: Ensure batch sizes are large enough to saturate GPU
