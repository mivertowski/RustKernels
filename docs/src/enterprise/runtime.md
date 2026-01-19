# Runtime

RustKernels 0.2.0 provides comprehensive runtime lifecycle management for production deployments.

## Overview

| Feature | Description |
|---------|-------------|
| **Lifecycle States** | Starting → Running → Draining → Stopped |
| **Configuration Presets** | Development, production, high-performance |
| **Graceful Shutdown** | Drain period with connection tracking |
| **Memory Management** | Pooling, pressure handling, reductions |

## Lifecycle States

```
┌──────────┐    start()    ┌─────────┐
│ Starting │ ────────────▶ │ Running │
└──────────┘               └────┬────┘
                                │ shutdown()
                                ▼
                          ┌──────────┐    drain complete    ┌─────────┐
                          │ Draining │ ──────────────────▶  │ Stopped │
                          └──────────┘                      └─────────┘
```

### State Descriptions

| State | Description |
|-------|-------------|
| `Starting` | Initializing kernels and resources |
| `Running` | Normal operation, accepting requests |
| `Draining` | Finishing in-flight requests, rejecting new ones |
| `Stopped` | All resources released |

## Runtime Configuration

### Presets

```rust
use rustkernel_core::runtime::{RuntimeConfig, RuntimePreset};

// Development: relaxed timeouts, verbose logging
let config = RuntimeConfig::development();

// Production: optimized for reliability
let config = RuntimeConfig::production();

// High-performance: maximum throughput
let config = RuntimeConfig::high_performance();
```

### Custom Configuration

```rust
let config = RuntimeConfig {
    gpu_enabled: true,
    max_kernel_instances: 100,
    worker_threads: 8,
    blocking_threads: 32,
    shutdown_timeout: Duration::from_secs(30),
    health_check_interval: Duration::from_secs(10),
    ..Default::default()
};
```

### Configuration Fields

| Field | Default | Description |
|-------|---------|-------------|
| `gpu_enabled` | true | Enable GPU backends |
| `max_kernel_instances` | 1000 | Maximum concurrent kernels |
| `worker_threads` | CPU count | Async worker threads |
| `blocking_threads` | 512 | Blocking task threads |
| `shutdown_timeout` | 30s | Graceful shutdown timeout |

## Runtime Builder

```rust
use rustkernel_core::runtime::RuntimeBuilder;

let runtime = RuntimeBuilder::new()
    .preset(RuntimePreset::Production)
    .with_max_instances(500)
    .with_shutdown_timeout(Duration::from_secs(60))
    .with_health_check_interval(Duration::from_secs(5))
    .build()?;

// Start the runtime
runtime.start().await?;

// Graceful shutdown
runtime.shutdown().await?;
```

## Graceful Shutdown

Handle shutdown signals properly:

```rust
use tokio::signal;

// Wait for shutdown signal
let ctrl_c = async {
    signal::ctrl_c().await.expect("failed to listen for ctrl+c");
};

tokio::select! {
    _ = ctrl_c => {
        println!("Shutdown signal received");
        runtime.shutdown().await?;
    }
    _ = server.run() => {}
}
```

### Drain Period

During draining:
1. New requests are rejected with 503 Service Unavailable
2. In-flight requests are allowed to complete
3. Health probes return "not ready"
4. After timeout, remaining requests are cancelled

## Memory Management

### Memory Configuration

```rust
use rustkernel_core::memory::MemoryConfig;

let config = MemoryConfig {
    max_gpu_memory: 8 * 1024 * 1024 * 1024, // 8GB
    max_staging_memory: 2 * 1024 * 1024 * 1024, // 2GB
    pooling_enabled: true,
    pressure_threshold: 0.8, // Warn at 80% usage
    auto_defrag: true,
    ..Default::default()
};
```

### Memory Pressure Levels

| Level | Threshold | Action |
|-------|-----------|--------|
| `Normal` | < 70% | Normal operation |
| `Warning` | 70-85% | Log warnings, defer allocations |
| `Critical` | 85-95% | Reject new kernels |
| `Emergency` | > 95% | Emergency cleanup |

### Size-Stratified Pooling

```rust
use rustkernel_core::memory::KernelMemoryManager;

let manager = KernelMemoryManager::new(config);

// Allocate from pool
let buffer = manager.allocate(1024 * 1024)?; // 1MB

// Return to pool
manager.deallocate(buffer);

// Check stats
let stats = manager.stats();
println!("Allocated: {} bytes", stats.allocated_bytes);
println!("Pool utilization: {:.1}%", stats.pool_utilization * 100.0);
```

## Production Configuration

### Unified Config

```rust
use rustkernel_core::config::{ProductionConfig, ProductionConfigBuilder};

// Load from environment
let config = ProductionConfig::from_env()?;

// Load from file
let config = ProductionConfig::from_file("config/production.toml")?;

// Use builder
let config = ProductionConfigBuilder::production()
    .service_name("my-kernel-service")
    .environment("production")
    .runtime(|r| r
        .max_kernel_instances(500)
        .shutdown_timeout(Duration::from_secs(60)))
    .memory(|m| m
        .max_gpu_memory(16 * 1024 * 1024 * 1024))
    .build()?;

// Validate
config.validate()?;
```

### TOML Configuration

```toml
# production.toml
environment = "production"
service_name = "rustkernels"
service_version = "0.2.0"

[runtime]
gpu_enabled = true
max_kernel_instances = 500
worker_threads = 16
shutdown_timeout_secs = 60

[memory]
max_gpu_memory = 17179869184  # 16GB
max_staging_memory = 4294967296  # 4GB
pooling_enabled = true
pressure_threshold = 0.8

[security]
rbac_enabled = true
audit_logging = true

[observability]
metrics_enabled = true
tracing_enabled = true
```

## CLI Commands

```bash
# Show runtime status
rustkernel runtime status

# Show current configuration
rustkernel runtime show

# Initialize with preset
rustkernel runtime init --preset production

# Generate config template
rustkernel config generate --preset production --output config.toml

# Validate config file
rustkernel config validate config.toml

# Show environment variables
rustkernel config env
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUSTKERNEL_ENV` | Preset (development, production, hp) |
| `RUSTKERNEL_SERVICE_NAME` | Service name |
| `RUSTKERNEL_GPU_ENABLED` | Enable GPU (true/false) |
| `RUSTKERNEL_MAX_INSTANCES` | Max kernel instances |
| `RUSTKERNEL_MAX_GPU_MEMORY_GB` | Max GPU memory in GB |

## Next Steps

- [Service Deployment](ecosystem.md) - Deploy as a service
- [Observability](observability.md) - Monitor runtime metrics
