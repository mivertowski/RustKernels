//! Graph Neural Network kernels.
//!
//! This module provides GPU-accelerated GNN algorithms:
//! - GNNInference - Message passing neural network inference
//! - GraphAttention - Graph attention network (GAT) layers

use crate::types::CsrGraph;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use serde::{Deserialize, Serialize};

// ============================================================================
// GNN Inference Kernel
// ============================================================================

/// Configuration for GNN inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNConfig {
    /// Number of message passing layers.
    pub num_layers: usize,
    /// Hidden dimension size.
    pub hidden_dim: usize,
    /// Output dimension (number of classes or embedding size).
    pub output_dim: usize,
    /// Aggregation function for messages.
    pub aggregation: AggregationType,
    /// Activation function.
    pub activation: ActivationType,
    /// Dropout rate (0-1).
    pub dropout: f64,
    /// Whether to add self-loops.
    pub add_self_loops: bool,
    /// Whether to use layer normalization.
    pub layer_norm: bool,
}

impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            output_dim: 32,
            aggregation: AggregationType::Mean,
            activation: ActivationType::ReLU,
            dropout: 0.0,
            add_self_loops: true,
            layer_norm: true,
        }
    }
}

/// Message aggregation type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregationType {
    /// Sum of neighbor messages.
    Sum,
    /// Mean of neighbor messages.
    Mean,
    /// Max pooling over neighbors.
    Max,
    /// GraphSAGE-style sample and aggregate.
    SAGE,
}

/// Activation function type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivationType {
    /// Rectified Linear Unit.
    ReLU,
    /// Leaky ReLU with alpha=0.01.
    LeakyReLU,
    /// Exponential Linear Unit.
    ELU,
    /// Sigmoid function.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// No activation.
    None,
}

/// GNN layer weights (simulated).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNWeights {
    /// Weight matrices per layer.
    pub layer_weights: Vec<Vec<Vec<f64>>>,
    /// Bias vectors per layer.
    pub layer_biases: Vec<Vec<f64>>,
}

impl GNNWeights {
    /// Create random weights for testing.
    pub fn random(input_dim: usize, config: &GNNConfig) -> Self {
        use rand::{Rng, rng};
        let mut r = rng();

        let mut layer_weights = Vec::new();
        let mut layer_biases = Vec::new();

        let mut prev_dim = input_dim;

        for i in 0..config.num_layers {
            let out_dim = if i == config.num_layers - 1 {
                config.output_dim
            } else {
                config.hidden_dim
            };

            // Xavier initialization
            let scale = (2.0 / (prev_dim + out_dim) as f64).sqrt();

            let weights: Vec<Vec<f64>> = (0..prev_dim)
                .map(|_| {
                    (0..out_dim)
                        .map(|_| r.random_range(-scale..scale))
                        .collect()
                })
                .collect();

            let biases: Vec<f64> = (0..out_dim).map(|_| 0.0).collect();

            layer_weights.push(weights);
            layer_biases.push(biases);
            prev_dim = out_dim;
        }

        Self {
            layer_weights,
            layer_biases,
        }
    }
}

/// Result of GNN inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNResult {
    /// Node embeddings after all layers.
    pub embeddings: Vec<Vec<f64>>,
    /// Class predictions (if classification).
    pub predictions: Option<Vec<usize>>,
    /// Softmax probabilities (if classification).
    pub probabilities: Option<Vec<Vec<f64>>>,
}

/// GNN Inference kernel.
///
/// Performs message passing neural network inference on graph data.
/// Supports various aggregation strategies and can be used for
/// node classification, link prediction, and graph-level tasks.
#[derive(Debug, Clone)]
pub struct GNNInference {
    metadata: KernelMetadata,
}

impl Default for GNNInference {
    fn default() -> Self {
        Self::new()
    }
}

impl GNNInference {
    /// Create a new GNN Inference kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/gnn-inference", Domain::GraphAnalytics)
                .with_description("Message passing neural network inference")
                .with_throughput(10_000)
                .with_latency_us(100.0),
        }
    }

    /// Run GNN inference on a graph.
    #[allow(clippy::needless_range_loop)]
    pub fn compute(
        graph: &CsrGraph,
        node_features: &[Vec<f64>],
        weights: &GNNWeights,
        config: &GNNConfig,
    ) -> GNNResult {
        if graph.num_nodes == 0 || node_features.is_empty() {
            return GNNResult {
                embeddings: Vec::new(),
                predictions: None,
                probabilities: None,
            };
        }

        let n = graph.num_nodes;

        // Build adjacency list from CSR format (with optional self-loops)
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for node in 0..n {
            let start = graph.row_offsets[node] as usize;
            let end = graph.row_offsets[node + 1] as usize;
            for &neighbor in &graph.col_indices[start..end] {
                adj[node].push(neighbor as usize);
                // Add reverse edge for undirected
                if !adj[neighbor as usize].contains(&node) {
                    adj[neighbor as usize].push(node);
                }
            }
        }

        if config.add_self_loops {
            for i in 0..n {
                if !adj[i].contains(&i) {
                    adj[i].push(i);
                }
            }
        }

        // Initialize embeddings from features
        let mut embeddings: Vec<Vec<f64>> = node_features.to_vec();

        // Run message passing layers
        for layer_idx in 0..config.num_layers {
            embeddings = Self::message_passing_layer(
                &embeddings,
                &adj,
                &weights.layer_weights[layer_idx],
                &weights.layer_biases[layer_idx],
                config,
                layer_idx == config.num_layers - 1,
            );
        }

        // Compute predictions if output looks like classification
        let (predictions, probabilities) = if config.output_dim > 1 {
            let probs: Vec<Vec<f64>> = embeddings.iter().map(|e| Self::softmax(e)).collect();
            let preds: Vec<usize> = probs
                .iter()
                .map(|p| {
                    p.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();
            (Some(preds), Some(probs))
        } else {
            (None, None)
        };

        GNNResult {
            embeddings,
            predictions,
            probabilities,
        }
    }

    /// Single message passing layer.
    #[allow(clippy::needless_range_loop)]
    fn message_passing_layer(
        embeddings: &[Vec<f64>],
        adj: &[Vec<usize>],
        weights: &[Vec<f64>],
        biases: &[f64],
        config: &GNNConfig,
        is_last: bool,
    ) -> Vec<Vec<f64>> {
        let n = embeddings.len();
        let out_dim = biases.len();
        let mut new_embeddings = vec![vec![0.0; out_dim]; n];

        for i in 0..n {
            // Aggregate neighbor messages
            let aggregated = Self::aggregate_neighbors(embeddings, &adj[i], config.aggregation);

            // Transform: out = activation(W * aggregated + b)
            for j in 0..out_dim {
                let mut val = biases[j];
                for (k, &agg_val) in aggregated.iter().enumerate() {
                    if k < weights.len() && j < weights[k].len() {
                        val += weights[k][j] * agg_val;
                    }
                }

                // Apply activation (except on last layer if doing classification)
                if !is_last {
                    val = Self::activate(val, config.activation);
                }

                new_embeddings[i][j] = val;
            }

            // Layer normalization
            if config.layer_norm && !is_last {
                let mean: f64 = new_embeddings[i].iter().sum::<f64>() / out_dim as f64;
                let var: f64 = new_embeddings[i]
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / out_dim as f64;
                let std = (var + 1e-5).sqrt();

                for j in 0..out_dim {
                    new_embeddings[i][j] = (new_embeddings[i][j] - mean) / std;
                }
            }
        }

        new_embeddings
    }

    /// Aggregate messages from neighbors.
    fn aggregate_neighbors(
        embeddings: &[Vec<f64>],
        neighbors: &[usize],
        agg_type: AggregationType,
    ) -> Vec<f64> {
        if neighbors.is_empty() {
            return vec![0.0; embeddings.first().map(|e| e.len()).unwrap_or(0)];
        }

        let dim = embeddings[neighbors[0]].len();

        match agg_type {
            AggregationType::Sum => {
                let mut result = vec![0.0; dim];
                for &n in neighbors {
                    for (i, &v) in embeddings[n].iter().enumerate() {
                        result[i] += v;
                    }
                }
                result
            }
            AggregationType::Mean => {
                let mut result = vec![0.0; dim];
                for &n in neighbors {
                    for (i, &v) in embeddings[n].iter().enumerate() {
                        result[i] += v;
                    }
                }
                let count = neighbors.len() as f64;
                result.iter_mut().for_each(|v| *v /= count);
                result
            }
            AggregationType::Max => {
                let mut result = vec![f64::NEG_INFINITY; dim];
                for &n in neighbors {
                    for (i, &v) in embeddings[n].iter().enumerate() {
                        result[i] = result[i].max(v);
                    }
                }
                result
            }
            AggregationType::SAGE => {
                // GraphSAGE: concat(self, mean(neighbors))
                // Simplified: just use mean here
                let mut result = vec![0.0; dim];
                for &n in neighbors {
                    for (i, &v) in embeddings[n].iter().enumerate() {
                        result[i] += v;
                    }
                }
                let count = neighbors.len() as f64;
                result.iter_mut().for_each(|v| *v /= count);
                result
            }
        }
    }

    /// Apply activation function.
    fn activate(x: f64, activation: ActivationType) -> f64 {
        match activation {
            ActivationType::ReLU => x.max(0.0),
            ActivationType::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            ActivationType::ELU => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::None => x,
        }
    }

    /// Softmax for classification.
    fn softmax(x: &[f64]) -> Vec<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = x.iter().map(|v| (v - max_val).exp()).sum();
        x.iter().map(|v| (v - max_val).exp() / exp_sum).collect()
    }
}

impl GpuKernel for GNNInference {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Graph Attention Kernel
// ============================================================================

/// Configuration for graph attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Hidden dimension per head.
    pub head_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Dropout for attention weights.
    pub attention_dropout: f64,
    /// Whether to concatenate heads or average.
    pub concat_heads: bool,
    /// Negative slope for LeakyReLU in attention.
    pub negative_slope: f64,
}

impl Default for GraphAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            head_dim: 16,
            output_dim: 64,
            attention_dropout: 0.0,
            concat_heads: true,
            negative_slope: 0.2,
        }
    }
}

/// Attention weights for GAT layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GATWeights {
    /// Query transformation weights per head.
    pub query_weights: Vec<Vec<Vec<f64>>>,
    /// Key transformation weights per head.
    pub key_weights: Vec<Vec<Vec<f64>>>,
    /// Value transformation weights per head.
    pub value_weights: Vec<Vec<Vec<f64>>>,
    /// Attention vector per head.
    pub attention_vectors: Vec<Vec<f64>>,
    /// Output projection weights.
    pub output_weights: Vec<Vec<f64>>,
}

impl GATWeights {
    /// Create random weights for testing.
    pub fn random(input_dim: usize, config: &GraphAttentionConfig) -> Self {
        use rand::{Rng, rng};
        let mut r = rng();

        let scale = (2.0 / (input_dim + config.head_dim) as f64).sqrt();

        let mut query_weights = Vec::new();
        let mut key_weights = Vec::new();
        let mut value_weights = Vec::new();
        let mut attention_vectors = Vec::new();

        for _ in 0..config.num_heads {
            let q: Vec<Vec<f64>> = (0..input_dim)
                .map(|_| {
                    (0..config.head_dim)
                        .map(|_| r.random_range(-scale..scale))
                        .collect()
                })
                .collect();
            let k: Vec<Vec<f64>> = (0..input_dim)
                .map(|_| {
                    (0..config.head_dim)
                        .map(|_| r.random_range(-scale..scale))
                        .collect()
                })
                .collect();
            let v: Vec<Vec<f64>> = (0..input_dim)
                .map(|_| {
                    (0..config.head_dim)
                        .map(|_| r.random_range(-scale..scale))
                        .collect()
                })
                .collect();
            let a: Vec<f64> = (0..config.head_dim * 2)
                .map(|_| r.random_range(-scale..scale))
                .collect();

            query_weights.push(q);
            key_weights.push(k);
            value_weights.push(v);
            attention_vectors.push(a);
        }

        let total_dim = if config.concat_heads {
            config.num_heads * config.head_dim
        } else {
            config.head_dim
        };

        let out_scale = (2.0 / (total_dim + config.output_dim) as f64).sqrt();
        let output_weights: Vec<Vec<f64>> = (0..total_dim)
            .map(|_| {
                (0..config.output_dim)
                    .map(|_| r.random_range(-out_scale..out_scale))
                    .collect()
            })
            .collect();

        Self {
            query_weights,
            key_weights,
            value_weights,
            attention_vectors,
            output_weights,
        }
    }
}

/// Result of graph attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GATResult {
    /// Output embeddings.
    pub embeddings: Vec<Vec<f64>>,
    /// Attention weights per head (source, target, weight).
    pub attention_weights: Vec<Vec<(usize, usize, f64)>>,
}

/// Graph Attention kernel.
///
/// Implements Graph Attention Networks (GAT) with multi-head attention.
/// Learns to weight neighbor contributions based on their relevance
/// to each node.
#[derive(Debug, Clone)]
pub struct GraphAttention {
    metadata: KernelMetadata,
}

impl Default for GraphAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAttention {
    /// Create a new Graph Attention kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("graph/graph-attention", Domain::GraphAnalytics)
                .with_description("Graph attention networks with multi-head attention")
                .with_throughput(5_000)
                .with_latency_us(200.0),
        }
    }

    /// Compute graph attention layer.
    #[allow(clippy::needless_range_loop)]
    pub fn compute(
        graph: &CsrGraph,
        node_features: &[Vec<f64>],
        weights: &GATWeights,
        config: &GraphAttentionConfig,
    ) -> GATResult {
        if graph.num_nodes == 0 || node_features.is_empty() {
            return GATResult {
                embeddings: Vec::new(),
                attention_weights: Vec::new(),
            };
        }

        let n = graph.num_nodes;

        // Build adjacency with self-loops from CSR format
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for node in 0..n {
            let start = graph.row_offsets[node] as usize;
            let end = graph.row_offsets[node + 1] as usize;
            for &neighbor in &graph.col_indices[start..end] {
                adj[node].push(neighbor as usize);
                if !adj[neighbor as usize].contains(&node) {
                    adj[neighbor as usize].push(node);
                }
            }
        }
        for i in 0..n {
            if !adj[i].contains(&i) {
                adj[i].push(i);
            }
        }

        // Compute attention for each head
        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut all_attention_weights: Vec<Vec<(usize, usize, f64)>> = Vec::new();

        for head in 0..config.num_heads {
            let (output, attn_weights) = Self::compute_head(
                node_features,
                &adj,
                &weights.query_weights[head],
                &weights.key_weights[head],
                &weights.value_weights[head],
                &weights.attention_vectors[head],
                config,
            );
            head_outputs.push(output);
            all_attention_weights.push(attn_weights);
        }

        // Combine heads
        let combined: Vec<Vec<f64>> = if config.concat_heads {
            (0..n)
                .map(|i| head_outputs.iter().flat_map(|h| h[i].clone()).collect())
                .collect()
        } else {
            // Average heads
            (0..n)
                .map(|i| {
                    let dim = head_outputs[0][i].len();
                    let mut avg = vec![0.0; dim];
                    for h in &head_outputs {
                        for (j, &v) in h[i].iter().enumerate() {
                            avg[j] += v;
                        }
                    }
                    avg.iter_mut().for_each(|v| *v /= config.num_heads as f64);
                    avg
                })
                .collect()
        };

        // Output projection
        let embeddings: Vec<Vec<f64>> = combined
            .iter()
            .map(|c| Self::linear_transform(c, &weights.output_weights))
            .collect();

        GATResult {
            embeddings,
            attention_weights: all_attention_weights,
        }
    }

    /// Compute single attention head.
    #[allow(clippy::type_complexity)]
    fn compute_head(
        features: &[Vec<f64>],
        adj: &[Vec<usize>],
        query_w: &[Vec<f64>],
        key_w: &[Vec<f64>],
        value_w: &[Vec<f64>],
        attn_vec: &[f64],
        config: &GraphAttentionConfig,
    ) -> (Vec<Vec<f64>>, Vec<(usize, usize, f64)>) {
        let n = features.len();
        let head_dim = config.head_dim;

        // Transform features to Q, K, V
        let queries: Vec<Vec<f64>> = features
            .iter()
            .map(|f| Self::linear_transform(f, query_w))
            .collect();
        let keys: Vec<Vec<f64>> = features
            .iter()
            .map(|f| Self::linear_transform(f, key_w))
            .collect();
        let values: Vec<Vec<f64>> = features
            .iter()
            .map(|f| Self::linear_transform(f, value_w))
            .collect();

        let mut output = vec![vec![0.0; head_dim]; n];
        let mut attention_list: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            if adj[i].is_empty() {
                continue;
            }

            // Compute attention scores for neighbors
            let mut scores: Vec<f64> = Vec::with_capacity(adj[i].len());

            for &j in &adj[i] {
                // Concatenate Q_i and K_j, apply attention vector
                let mut concat = queries[i].clone();
                concat.extend(keys[j].iter().cloned());

                let score: f64 = concat.iter().zip(attn_vec.iter()).map(|(c, a)| c * a).sum();

                // LeakyReLU
                let score = if score > 0.0 {
                    score
                } else {
                    config.negative_slope * score
                };

                scores.push(score);
            }

            // Softmax over neighbors
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let attention: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();

            // Aggregate values weighted by attention
            for (idx, &j) in adj[i].iter().enumerate() {
                let attn = attention[idx];
                attention_list.push((i, j, attn));

                for (k, &v) in values[j].iter().enumerate() {
                    output[i][k] += attn * v;
                }
            }
        }

        (output, attention_list)
    }

    /// Linear transformation.
    fn linear_transform(input: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        if weights.is_empty() {
            return Vec::new();
        }

        let out_dim = weights[0].len();
        let mut output = vec![0.0; out_dim];

        for (i, &x) in input.iter().enumerate() {
            if i < weights.len() {
                for (j, &w) in weights[i].iter().enumerate() {
                    output[j] += x * w;
                }
            }
        }

        output
    }

    /// Get node importance based on attention received.
    pub fn node_importance(attention_weights: &[(usize, usize, f64)], n: usize) -> Vec<f64> {
        let mut importance = vec![0.0; n];
        let mut counts = vec![0usize; n];

        for &(_, target, weight) in attention_weights {
            if target < n {
                importance[target] += weight;
                counts[target] += 1;
            }
        }

        // Normalize by count
        for i in 0..n {
            if counts[i] > 0 {
                importance[i] /= counts[i] as f64;
            }
        }

        importance
    }
}

impl GpuKernel for GraphAttention {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_graph() -> CsrGraph {
        // Simple triangle graph: 0 -- 1 -- 2 -- 0
        CsrGraph::from_edges(3, &[(0, 1), (1, 2), (2, 0)])
    }

    #[test]
    fn test_gnn_inference_metadata() {
        let kernel = GNNInference::new();
        assert_eq!(kernel.metadata().id, "graph/gnn-inference");
    }

    #[test]
    fn test_gnn_inference_basic() {
        let graph = create_test_graph();
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let config = GNNConfig {
            num_layers: 2,
            hidden_dim: 4,
            output_dim: 2,
            ..Default::default()
        };

        let weights = GNNWeights::random(2, &config);
        let result = GNNInference::compute(&graph, &features, &weights, &config);

        assert_eq!(result.embeddings.len(), 3);
        assert_eq!(result.embeddings[0].len(), 2);
        assert!(result.predictions.is_some());
    }

    #[test]
    fn test_gnn_aggregation_types() {
        let graph = create_test_graph();
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        for agg in [
            AggregationType::Sum,
            AggregationType::Mean,
            AggregationType::Max,
            AggregationType::SAGE,
        ] {
            let config = GNNConfig {
                aggregation: agg,
                num_layers: 1,
                hidden_dim: 4,
                output_dim: 2,
                ..Default::default()
            };

            let weights = GNNWeights::random(2, &config);
            let result = GNNInference::compute(&graph, &features, &weights, &config);

            assert_eq!(result.embeddings.len(), 3);
        }
    }

    #[test]
    fn test_gnn_empty_graph() {
        let graph = CsrGraph::empty();
        let features: Vec<Vec<f64>> = vec![];
        let config = GNNConfig::default();
        let weights = GNNWeights::random(2, &config);

        let result = GNNInference::compute(&graph, &features, &weights, &config);
        assert!(result.embeddings.is_empty());
    }

    #[test]
    fn test_graph_attention_metadata() {
        let kernel = GraphAttention::new();
        assert_eq!(kernel.metadata().id, "graph/graph-attention");
    }

    #[test]
    fn test_graph_attention_basic() {
        let graph = create_test_graph();
        let features = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let config = GraphAttentionConfig {
            num_heads: 2,
            head_dim: 4,
            output_dim: 3,
            ..Default::default()
        };

        let weights = GATWeights::random(4, &config);
        let result = GraphAttention::compute(&graph, &features, &weights, &config);

        assert_eq!(result.embeddings.len(), 3);
        assert_eq!(result.embeddings[0].len(), 3);
        assert!(!result.attention_weights.is_empty());
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let graph = create_test_graph();
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let config = GraphAttentionConfig {
            num_heads: 1,
            head_dim: 4,
            output_dim: 2,
            ..Default::default()
        };

        let weights = GATWeights::random(2, &config);
        let result = GraphAttention::compute(&graph, &features, &weights, &config);

        // Group attention weights by source node
        let mut sums: HashMap<usize, f64> = HashMap::new();
        for &(src, _, weight) in &result.attention_weights[0] {
            *sums.entry(src).or_insert(0.0) += weight;
        }

        // Each source's attention should sum to ~1
        for (_, sum) in sums {
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Attention should sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_node_importance() {
        let attn_weights = vec![
            (0, 1, 0.5),
            (0, 2, 0.5),
            (1, 0, 0.3),
            (1, 2, 0.7),
            (2, 0, 0.4),
            (2, 1, 0.6),
        ];

        let importance = GraphAttention::node_importance(&attn_weights, 3);

        assert_eq!(importance.len(), 3);
        // Node 2 receives more attention on average
        assert!(importance.iter().all(|&i| i >= 0.0));
    }

    #[test]
    fn test_activation_functions() {
        assert_eq!(GNNInference::activate(1.0, ActivationType::ReLU), 1.0);
        assert_eq!(GNNInference::activate(-1.0, ActivationType::ReLU), 0.0);
        assert!((GNNInference::activate(0.0, ActivationType::Sigmoid) - 0.5).abs() < 0.001);
        assert_eq!(GNNInference::activate(1.0, ActivationType::None), 1.0);
    }
}
