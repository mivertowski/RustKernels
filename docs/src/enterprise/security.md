# Security

RustKernels 0.2.0 includes comprehensive enterprise security features for production deployments.

## Overview

| Feature | Description |
|---------|-------------|
| **Authentication** | JWT and API key validation |
| **Authorization** | Role-based access control (RBAC) |
| **Multi-tenancy** | Tenant isolation with resource quotas |
| **Secrets Management** | Secure credential storage abstraction |

## Authentication

Configure authentication using `AuthConfig`:

```rust
use rustkernel_core::security::{AuthConfig, AuthMethod};

let auth = AuthConfig {
    method: AuthMethod::Jwt {
        secret: "your-jwt-secret".to_string(),
        issuer: Some("rustkernels".to_string()),
        audience: None,
    },
    token_expiry: Duration::from_secs(3600),
    refresh_enabled: true,
};
```

### Supported Methods

- **JWT**: JSON Web Token validation with configurable issuer/audience
- **API Key**: Simple API key authentication for service-to-service calls
- **None**: Disabled authentication (development only)

## Role-Based Access Control

Define permissions for kernel operations:

```rust
use rustkernel_core::security::{Role, KernelPermission, PermissionSet};

// Built-in roles
let executor = Role::KernelExecutor;  // Can execute kernels
let admin = Role::Admin;              // Full access

// Custom permissions
let permissions = PermissionSet::new()
    .with_permission(KernelPermission::Execute)
    .with_permission(KernelPermission::Monitor)
    .for_domains(vec![Domain::GraphAnalytics, Domain::Compliance]);
```

### Permission Types

| Permission | Description |
|------------|-------------|
| `Execute` | Run kernel computations |
| `Configure` | Modify kernel configuration |
| `Monitor` | View metrics and health status |
| `Admin` | Full administrative access |

## Security Context

Pass security context through kernel execution:

```rust
use rustkernel_core::security::SecurityContext;

let ctx = SecurityContext::new(user_id, tenant_id)
    .with_roles(vec![Role::KernelExecutor])
    .with_permissions(vec![KernelPermission::Execute]);

// Execute with context
kernel.execute_with_context(&ctx, input).await?;
```

## Multi-Tenancy

Isolate kernels and resources by tenant:

```rust
use rustkernel_core::security::{TenantId, TenantConfig};

let tenant = TenantConfig {
    id: TenantId::new("tenant-123"),
    name: "Acme Corp".to_string(),
    max_kernel_instances: 100,
    max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
    allowed_domains: vec![Domain::GraphAnalytics, Domain::Compliance],
};
```

## Secrets Management

Integrate with external secret stores:

```rust
use rustkernel_core::security::{SecretStore, SecretRef};

// Reference a secret
let api_key = SecretRef::new("external-api-key");

// Retrieve at runtime
let value = secret_store.get(&api_key).await?;
```

## Production Configuration

Enable security in production:

```rust
use rustkernel_core::config::ProductionConfig;

let config = ProductionConfig::production();
// Security is enabled by default in production preset:
// - RBAC enabled
// - Audit logging enabled
// - Multi-tenancy infrastructure ready

config.validate()?; // Warns if security is weak
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUSTKERNEL_AUTH_ENABLED` | Enable authentication/RBAC |
| `RUSTKERNEL_MULTI_TENANT` | Enable multi-tenancy |
| `RUSTKERNEL_JWT_SECRET` | JWT signing secret |

## Best Practices

1. **Always enable auth in production**: Use `ProductionConfig::production()`
2. **Principle of least privilege**: Grant minimum required permissions
3. **Rotate secrets regularly**: Use external secret management
4. **Enable audit logging**: Track all security events
5. **Validate tenant boundaries**: Test multi-tenant isolation

## Next Steps

- [Observability](observability.md) - Monitor security events
- [Runtime](runtime.md) - Configure production runtime
