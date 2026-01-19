# Resilience

RustKernels 0.2.0 includes production-grade resilience patterns for fault-tolerant deployments.

## Overview

| Pattern | Description |
|---------|-------------|
| **Circuit Breaker** | Prevent cascade failures |
| **Retry** | Automatic retry with backoff |
| **Timeout** | Deadline propagation |
| **Health Checks** | Liveness/readiness probes |
| **Recovery** | Failure recovery strategies |

## Circuit Breaker

Protect kernels from repeated failures:

```rust
use rustkernel_core::resilience::{CircuitBreaker, CircuitBreakerConfig};

let config = CircuitBreakerConfig {
    failure_threshold: 5,      // Open after 5 failures
    success_threshold: 2,      // Close after 2 successes
    timeout: Duration::from_secs(30), // Half-open after 30s
    ..Default::default()
};

let cb = CircuitBreaker::new(config);

// Execute with protection
match cb.call(|| kernel.execute(input)).await {
    Ok(result) => handle_result(result),
    Err(ResilienceError::CircuitOpen) => use_fallback(),
    Err(e) => handle_error(e),
}
```

### Circuit States

| State | Description |
|-------|-------------|
| `Closed` | Normal operation, requests pass through |
| `Open` | Failures exceeded threshold, requests rejected |
| `HalfOpen` | Testing recovery, limited requests allowed |

### Monitoring

```rust
let state = cb.state();
let stats = cb.stats();

println!("State: {:?}", state);
println!("Failures: {}", stats.failure_count);
println!("Success rate: {:.1}%", stats.success_rate * 100.0);
```

## Retry with Backoff

Automatically retry transient failures:

```rust
use rustkernel_core::resilience::{RetryConfig, BackoffStrategy};

let config = RetryConfig {
    max_attempts: 3,
    backoff: BackoffStrategy::ExponentialWithJitter {
        initial: Duration::from_millis(100),
        max: Duration::from_secs(10),
        multiplier: 2.0,
    },
    retryable_errors: vec![ErrorKind::Transient, ErrorKind::Timeout],
};

let result = config.execute(|| kernel.execute(input)).await?;
```

### Backoff Strategies

| Strategy | Description |
|----------|-------------|
| `Fixed` | Constant delay between retries |
| `Linear` | Linearly increasing delay |
| `Exponential` | Exponentially increasing delay |
| `ExponentialWithJitter` | Exponential with random jitter (recommended) |

## Timeout and Deadlines

### Per-Kernel Timeouts

```rust
use rustkernel_core::resilience::TimeoutConfig;

let config = TimeoutConfig {
    default_timeout: Duration::from_secs(30),
    max_timeout: Duration::from_secs(300),
    propagate_deadline: true,
    include_queue_time: true,
};
```

### Deadline Context

Propagate deadlines through K2K chains:

```rust
use rustkernel_core::resilience::DeadlineContext;

let deadline = DeadlineContext::new(Duration::from_secs(10));

// Check remaining time
if deadline.remaining() < Duration::from_secs(1) {
    return Err(DeadlineExceeded);
}

// Execute with deadline
let result = deadline.execute(|| kernel.execute(input)).await?;

// Create child deadline for downstream calls
let child = deadline.child_with_timeout(Duration::from_secs(5));
```

## Health Checks

### Health Probe Configuration

```rust
use rustkernel_core::resilience::{HealthProbe, HealthProbeConfig};

let probe = HealthProbe::new(HealthProbeConfig {
    interval: Duration::from_secs(10),
    timeout: Duration::from_secs(5),
    failure_threshold: 3,
    success_threshold: 1,
});

// Register kernel for monitoring
probe.register("graph/pagerank", kernel.clone());

// Get health status
let status = probe.check("graph/pagerank").await;
```

### Health Status

```rust
use rustkernel_core::resilience::{HealthCheckResult, HealthStatus};

let result = HealthCheckResult::healthy();
let result = HealthCheckResult::degraded("High latency");
let result = HealthCheckResult::unhealthy("GPU unavailable");

// Check status
if result.status == HealthStatus::Unhealthy {
    take_kernel_offline();
}
```

### Kubernetes Probes

```rust
// Liveness: Is the service alive?
app.route("/health/live", get(|| async { "OK" }));

// Readiness: Can the service handle traffic?
app.route("/health/ready", get(|| async {
    if all_kernels_ready() { "OK" } else { StatusCode::SERVICE_UNAVAILABLE }
}));

// Startup: Has the service finished initializing?
app.route("/health/startup", get(|| async {
    if initialization_complete() { "OK" } else { StatusCode::SERVICE_UNAVAILABLE }
}));
```

## Recovery Policies

### Recovery Strategies

```rust
use rustkernel_core::resilience::{RecoveryPolicy, RecoveryStrategy};

let policy = RecoveryPolicy {
    strategy: RecoveryStrategy::RestartKernel,
    max_restarts: 3,
    restart_delay: Duration::from_secs(5),
    escalation: Some(RecoveryStrategy::NotifyOperator),
};
```

| Strategy | Description |
|----------|-------------|
| `RestartKernel` | Restart the failed kernel |
| `UseReplica` | Switch to a replica |
| `Degrade` | Continue with reduced functionality |
| `Failover` | Switch to backup system |
| `NotifyOperator` | Alert human operator |

### Checkpointing

For long-running kernels, implement checkpointing:

```rust
use rustkernel_core::traits::CheckpointableKernel;

impl CheckpointableKernel for MyKernel {
    type Checkpoint = MyCheckpoint;

    async fn checkpoint(&self) -> Result<Self::Checkpoint> {
        Ok(MyCheckpoint { state: self.state.clone() })
    }

    async fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<()> {
        self.state = checkpoint.state;
        Ok(())
    }
}
```

## Production Configuration

```rust
use rustkernel_core::resilience::ResilienceConfig;

// Production preset with sensible defaults
let config = ResilienceConfig::production();

// Customize
let config = ResilienceConfig {
    circuit_breaker: CircuitBreakerConfig::production(),
    retry: RetryConfig::production(),
    timeout: TimeoutConfig::production(),
    health_check: HealthCheckConfig::production(),
    ..Default::default()
};
```

## CLI Commands

```bash
# Check health status
rustkernel health

# JSON output for monitoring
rustkernel health --format json

# Check specific component
rustkernel health --component runtime
```

## Next Steps

- [Runtime](runtime.md) - Lifecycle management
- [Observability](observability.md) - Monitor resilience metrics
