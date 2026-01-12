//! Test kernels for validation and benchmarking.
//!
//! This module provides simple kernels for testing the kernel framework:
//! - `VectorAdd`: Batch kernel for vector addition
//! - `EchoKernel`: Ring kernel for message echo (latency testing)

use crate::domain::Domain;
use crate::error::Result;
use crate::kernel::KernelMetadata;
use crate::traits::{BatchKernel, GpuKernel};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ============================================================================
// VectorAdd Batch Kernel
// ============================================================================

/// Input for vector addition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorAddInput {
    /// First vector.
    pub a: Vec<f32>,
    /// Second vector.
    pub b: Vec<f32>,
}

/// Output from vector addition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorAddOutput {
    /// Result vector (a + b).
    pub result: Vec<f32>,
}

/// Simple vector addition kernel.
///
/// This is a batch kernel that adds two vectors element-wise.
/// Used for testing and validation of the kernel framework.
#[derive(Debug, Clone)]
pub struct VectorAdd {
    metadata: KernelMetadata,
}

impl VectorAdd {
    /// Create a new VectorAdd kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("core/vector-add", Domain::Core)
                .with_description("Element-wise vector addition")
                .with_throughput(10_000_000)
                .with_latency_us(10.0),
        }
    }

    /// Perform vector addition (CPU implementation).
    fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
}

impl Default for VectorAdd {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for VectorAdd {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }

    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl BatchKernel<VectorAddInput, VectorAddOutput> for VectorAdd {
    async fn execute(&self, input: VectorAddInput) -> Result<VectorAddOutput> {
        self.validate_input(&input)?;
        let result = Self::add_vectors(&input.a, &input.b);
        Ok(VectorAddOutput { result })
    }

    fn validate_input(&self, input: &VectorAddInput) -> Result<()> {
        if input.a.len() != input.b.len() {
            return Err(crate::error::KernelError::validation(format!(
                "Vector lengths must match: a.len()={}, b.len()={}",
                input.a.len(),
                input.b.len()
            )));
        }
        Ok(())
    }
}

// ============================================================================
// Echo Ring Kernel
// ============================================================================

/// Echo request message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoRequest {
    /// Message to echo back.
    pub message: String,
    /// Sequence number for ordering.
    pub sequence: u64,
}

/// Echo response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoResponse {
    /// Echoed message.
    pub message: String,
    /// Original sequence number.
    pub sequence: u64,
    /// Timestamp of processing (nanoseconds since kernel start).
    pub processed_ns: u64,
}

/// Echo kernel state.
#[derive(Debug, Clone, Default)]
pub struct EchoState {
    /// Number of messages processed.
    pub messages_processed: u64,
    /// Start time for latency measurement.
    pub start_ns: u64,
}

/// Simple echo kernel for latency testing.
///
/// This is a ring kernel that echoes back messages with timing information.
/// Used for testing message round-trip latency.
#[derive(Debug, Clone)]
pub struct EchoKernel {
    metadata: KernelMetadata,
}

impl EchoKernel {
    /// Create a new EchoKernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("core/echo", Domain::Core)
                .with_description("Message echo for latency testing")
                .with_throughput(1_000_000)
                .with_latency_us(0.5),
        }
    }

    /// Process an echo request.
    pub fn process(state: &mut EchoState, request: EchoRequest) -> EchoResponse {
        state.messages_processed += 1;

        // Simple timestamp (would use HLC in real implementation)
        let processed_ns = state.messages_processed * 100; // Placeholder

        EchoResponse {
            message: request.message,
            sequence: request.sequence,
            processed_ns,
        }
    }

    /// Initialize state.
    pub fn initialize() -> EchoState {
        EchoState {
            messages_processed: 0,
            start_ns: 0,
        }
    }
}

impl Default for EchoKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for EchoKernel {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Matrix Multiply Batch Kernel (for benchmarking)
// ============================================================================

/// Input for matrix multiplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMulInput {
    /// First matrix (row-major, dimensions rows_a x cols_a).
    pub a: Vec<f32>,
    /// Second matrix (row-major, dimensions cols_a x cols_b).
    pub b: Vec<f32>,
    /// Rows in matrix A.
    pub rows_a: usize,
    /// Columns in matrix A (= rows in matrix B).
    pub cols_a: usize,
    /// Columns in matrix B.
    pub cols_b: usize,
}

/// Output from matrix multiplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMulOutput {
    /// Result matrix (row-major, dimensions rows_a x cols_b).
    pub result: Vec<f32>,
}

/// Matrix multiplication kernel.
///
/// This is a batch kernel that multiplies two matrices.
/// Used for benchmarking compute throughput.
#[derive(Debug, Clone)]
pub struct MatMul {
    metadata: KernelMetadata,
}

impl MatMul {
    /// Create a new MatMul kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("core/matmul", Domain::Core)
                .with_description("Matrix multiplication (GEMM)")
                .with_throughput(1_000_000)
                .with_latency_us(50.0)
                .with_gpu_native(true),
        }
    }

    /// Perform matrix multiplication (naive CPU implementation).
    fn matmul(a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; rows_a * cols_b];

        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0.0f32;
                for k in 0..cols_a {
                    sum += a[i * cols_a + k] * b[k * cols_b + j];
                }
                result[i * cols_b + j] = sum;
            }
        }

        result
    }
}

impl Default for MatMul {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for MatMul {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<MatMulInput, MatMulOutput> for MatMul {
    async fn execute(&self, input: MatMulInput) -> Result<MatMulOutput> {
        self.validate_input(&input)?;
        let result = Self::matmul(&input.a, &input.b, input.rows_a, input.cols_a, input.cols_b);
        Ok(MatMulOutput { result })
    }

    fn validate_input(&self, input: &MatMulInput) -> Result<()> {
        let expected_a = input.rows_a * input.cols_a;
        let expected_b = input.cols_a * input.cols_b;

        if input.a.len() != expected_a {
            return Err(crate::error::KernelError::validation(format!(
                "Matrix A size mismatch: expected {}, got {}",
                expected_a,
                input.a.len()
            )));
        }

        if input.b.len() != expected_b {
            return Err(crate::error::KernelError::validation(format!(
                "Matrix B size mismatch: expected {}, got {}",
                expected_b,
                input.b.len()
            )));
        }

        Ok(())
    }
}

// ============================================================================
// Reduce Sum Kernel
// ============================================================================

/// Input for sum reduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceSumInput {
    /// Data to sum.
    pub data: Vec<f32>,
}

/// Output from sum reduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReduceSumOutput {
    /// Sum of all elements.
    pub sum: f64,
    /// Count of elements.
    pub count: usize,
}

/// Sum reduction kernel.
///
/// Reduces a vector to its sum. Used for testing parallel reduction patterns.
#[derive(Debug, Clone)]
pub struct ReduceSum {
    metadata: KernelMetadata,
}

impl ReduceSum {
    /// Create a new ReduceSum kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("core/reduce-sum", Domain::Core)
                .with_description("Parallel sum reduction")
                .with_throughput(100_000_000)
                .with_latency_us(5.0),
        }
    }
}

impl Default for ReduceSum {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuKernel for ReduceSum {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<ReduceSumInput, ReduceSumOutput> for ReduceSum {
    async fn execute(&self, input: ReduceSumInput) -> Result<ReduceSumOutput> {
        let sum: f64 = input.data.iter().map(|&x| f64::from(x)).sum();
        Ok(ReduceSumOutput {
            sum,
            count: input.data.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::KernelMode;

    #[tokio::test]
    async fn test_vector_add() {
        let kernel = VectorAdd::new();
        assert_eq!(kernel.metadata().id, "core/vector-add");
        assert_eq!(kernel.metadata().mode, KernelMode::Batch);

        let input = VectorAddInput {
            a: vec![1.0, 2.0, 3.0],
            b: vec![4.0, 5.0, 6.0],
        };

        let output = kernel.execute(input).await.unwrap();
        assert_eq!(output.result, vec![5.0, 7.0, 9.0]);
    }

    #[tokio::test]
    async fn test_vector_add_validation() {
        let kernel = VectorAdd::new();

        let input = VectorAddInput {
            a: vec![1.0, 2.0],
            b: vec![1.0, 2.0, 3.0],
        };

        let result = kernel.execute(input).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_echo_kernel() {
        let kernel = EchoKernel::new();
        assert_eq!(kernel.metadata().id, "core/echo");
        assert_eq!(kernel.metadata().mode, KernelMode::Ring);

        let mut state = EchoKernel::initialize();
        let request = EchoRequest {
            message: "Hello".to_string(),
            sequence: 1,
        };

        let response = EchoKernel::process(&mut state, request);
        assert_eq!(response.message, "Hello");
        assert_eq!(response.sequence, 1);
        assert_eq!(state.messages_processed, 1);
    }

    #[tokio::test]
    async fn test_matmul() {
        let kernel = MatMul::new();

        // 2x2 * 2x2 matrix multiplication
        let input = MatMulInput {
            a: vec![1.0, 2.0, 3.0, 4.0],
            b: vec![5.0, 6.0, 7.0, 8.0],
            rows_a: 2,
            cols_a: 2,
            cols_b: 2,
        };

        let output = kernel.execute(input).await.unwrap();
        // [1,2] * [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        assert_eq!(output.result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[tokio::test]
    async fn test_reduce_sum() {
        let kernel = ReduceSum::new();

        let input = ReduceSumInput {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };

        let output = kernel.execute(input).await.unwrap();
        assert!((output.sum - 15.0).abs() < 1e-6);
        assert_eq!(output.count, 5);
    }
}
