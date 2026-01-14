# rustkernel-ml

[![Crates.io](https://img.shields.io/crates/v/rustkernel-ml.svg)](https://crates.io/crates/rustkernel-ml)
[![Documentation](https://docs.rs/rustkernel-ml/badge.svg)](https://docs.rs/rustkernel-ml)
[![License](https://img.shields.io/crates/l/rustkernel-ml.svg)](LICENSE)

GPU-accelerated machine learning kernels for clustering, anomaly detection, regression, and explainability.

## Kernels (17)

### Clustering (3 kernels)
- **KMeans** - Lloyd's algorithm with K-Means++ initialization
- **DBSCAN** - Density-based clustering with GPU union-find
- **HierarchicalClustering** - Agglomerative clustering

### Anomaly Detection (4 kernels)
- **IsolationForest** - Ensemble of isolation trees
- **LocalOutlierFactor** - k-NN density estimation
- **StreamingIsolationForest** - Online anomaly detection
- **AdaptiveThreshold** - Self-adjusting thresholds with drift detection

### Regression (2 kernels)
- **LinearRegression** - OLS with regularization
- **LogisticRegression** - Binary/multinomial classification

### NLP & Embeddings (2 kernels)
- **EmbeddingGeneration** - Hash-based text embeddings
- **SemanticSimilarity** - Multi-metric similarity search

### Healthcare (2 kernels)
- **DrugInteractionPrediction** - Multi-drug interaction prediction
- **ClinicalPathwayConformance** - Treatment guideline checking

### Federated Learning (1 kernel)
- **SecureAggregation** - Differential privacy aggregation

### Explainability (2 kernels)
- **SHAPValues** - Kernel SHAP for feature explanations
- **FeatureImportance** - Permutation-based importance

### Dimensionality Reduction (1 kernel)
- **PCA** - Principal component analysis

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-ml = "0.1.0"
```

## Usage

```rust
use rustkernel_ml::prelude::*;

// Create a KMeans kernel
let kmeans = KMeans::new();

// Cluster data points
let result = kmeans.cluster(&data, k, max_iterations);
```

## License

Apache-2.0
