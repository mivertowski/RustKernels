# Observability

RustKernels 0.2.0 provides comprehensive observability for production monitoring.

## Overview

| Feature | Description |
|---------|-------------|
| **Metrics** | Prometheus-compatible metrics export |
| **Tracing** | Distributed tracing with OTLP support |
| **Logging** | Structured logging with context propagation |
| **Alerting** | SLO-based alerts with multiple channels |

## Metrics

### Configuration

```rust
use rustkernel_core::observability::{MetricsConfig, ObservabilityConfig};

let config = ObservabilityConfig::production()
    .with_metrics(MetricsConfig {
        enabled: true,
        endpoint: "/metrics".to_string(),
        include_runtime: true,
        include_kernel_stats: true,
        histogram_buckets: vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    });
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rustkernel_executions_total` | Counter | Total kernel executions |
| `rustkernel_execution_duration_seconds` | Histogram | Execution latency |
| `rustkernel_active_kernels` | Gauge | Currently running kernels |
| `rustkernel_memory_bytes` | Gauge | Memory usage by pool |
| `rustkernel_circuit_breaker_state` | Gauge | Circuit breaker status |

### Custom Metrics

```rust
use rustkernel_core::observability::KernelMetrics;

let metrics = KernelMetrics::new("my-kernel");
metrics.record_execution(Duration::from_micros(150));
metrics.record_memory_allocated(1024 * 1024);
```

## Distributed Tracing

### OTLP Export

```rust
use rustkernel_core::observability::{TracingConfig, OtlpConfig};

let config = TracingConfig {
    enabled: true,
    sampling_rate: 0.1, // Sample 10% of requests
    otlp: Some(OtlpConfig {
        endpoint: "http://localhost:4317".to_string(),
        ..Default::default()
    }),
};
```

### Kernel Spans

Traces are automatically created for kernel executions:

```
[kernel:graph/pagerank] 15.2ms
├── [validate] 0.1ms
├── [prepare_input] 2.1ms
├── [gpu_execute] 12.5ms
└── [collect_output] 0.5ms
```

### Context Propagation

Trace context propagates through K2K messages:

```rust
use rustkernel_core::observability::TraceContext;

// Context automatically propagated
let result = kernel_a.execute_with_context(&ctx, input).await?;
// Child kernel inherits trace context
```

## Structured Logging

### Configuration

```rust
use rustkernel_core::observability::{LoggingConfig, LogLevel};

let config = LoggingConfig {
    level: LogLevel::Info,
    format: LogFormat::Json,
    include_kernel_context: true,
    per_domain_levels: vec![
        (Domain::Compliance, LogLevel::Debug), // More verbose for compliance
    ],
};
```

### Log Output

```json
{
  "timestamp": "2026-01-19T10:30:00Z",
  "level": "INFO",
  "message": "Kernel execution complete",
  "kernel_id": "graph/pagerank",
  "domain": "GraphAnalytics",
  "duration_ms": 15.2,
  "trace_id": "abc123",
  "tenant_id": "tenant-456"
}
```

## Alerting

### Alert Rules

```rust
use rustkernel_core::observability::{AlertRule, AlertSeverity, AlertCondition};

let rule = AlertRule {
    name: "high_latency".to_string(),
    condition: AlertCondition::LatencyExceeds {
        threshold: Duration::from_millis(100),
        percentile: 95,
    },
    severity: AlertSeverity::Warning,
    for_duration: Duration::from_secs(60),
};
```

### Notification Channels

```rust
use rustkernel_core::observability::{AlertChannel, SlackConfig};

let channels = vec![
    AlertChannel::Slack(SlackConfig {
        webhook_url: "https://hooks.slack.com/...".to_string(),
        channel: Some("#alerts".to_string()),
    }),
    AlertChannel::PagerDuty(PagerDutyConfig {
        service_key: "...".to_string(),
    }),
];
```

### SLO Monitoring

```rust
use rustkernel_core::slo::{SLOValidator, SLOTarget};

let slo = SLOTarget {
    latency_p99: Duration::from_millis(50),
    availability: 0.999,
    error_rate: 0.001,
};

let validator = SLOValidator::new(slo);
let result = validator.check(&metrics)?;
if !result.compliant {
    alert_slo_breach(&result);
}
```

## Production Setup

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rustkernels'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import the provided dashboard for:
- Kernel execution rates
- Latency percentiles
- Memory usage
- Circuit breaker states
- Error rates by domain

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUSTKERNEL_METRICS_ENABLED` | Enable metrics export |
| `RUSTKERNEL_TRACING_ENABLED` | Enable distributed tracing |
| `RUSTKERNEL_OTLP_ENDPOINT` | OTLP collector endpoint |
| `RUSTKERNEL_LOG_LEVEL` | Default log level |

## Next Steps

- [Resilience](resilience.md) - Monitor circuit breaker health
- [Runtime](runtime.md) - Configure health endpoints
