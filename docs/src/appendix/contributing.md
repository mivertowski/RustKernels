# Contributing

Thank you for your interest in contributing to RustKernels!

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mivertowski/RustKernels.git
   cd RustKernels
   ```

2. **Ensure RustCompute is available**:
   ```bash
   # Clone alongside RustKernels
   cd ..
   git clone https://github.com/mivertowski/RustCompute.git
   ```

3. **Build and test**:
   ```bash
   cd RustKernels
   cargo build --workspace
   cargo test --workspace
   ```

### Development Commands

```bash
# Format code
cargo fmt --all

# Lint
cargo clippy --all-targets --all-features -- -D warnings

# Run specific domain tests
cargo test --package rustkernel-graph

# Build documentation
cargo doc --workspace --no-deps --open

# Check all features compile
cargo check --all-features
```

## Contributing Code

### Types of Contributions

- **Bug fixes**: Fix issues in existing kernels
- **New kernels**: Add kernels to existing domains
- **Documentation**: Improve docs, add examples
- **Performance**: Optimize existing implementations
- **Tests**: Increase test coverage

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**
4. **Run tests**: `cargo test --workspace`
5. **Run lints**: `cargo clippy --all-targets -- -D warnings`
6. **Format code**: `cargo fmt --all`
7. **Commit with clear messages**
8. **Open a Pull Request**

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(graph): add triangle counting kernel
fix(accounting): correct VAT detection for reduced rates
docs(readme): update kernel count
test(ml): add edge cases for kmeans
```

## Adding a New Kernel

### 1. Define the Kernel Struct

```rust
use rustkernel_core::{GpuKernel, KernelMetadata, Domain, KernelMode};

#[derive(Debug, Clone)]
pub struct MyNewKernel {
    metadata: KernelMetadata,
}

impl MyNewKernel {
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata {
                id: "domain/my-new-kernel".to_string(),
                mode: KernelMode::Batch,
                domain: Domain::MyDomain,
                description: "Description of what this kernel does".to_string(),
                expected_throughput: 100_000,
                target_latency_us: 50.0,
                requires_gpu_native: false,
                version: 1,
            },
        }
    }
}

impl GpuKernel for MyNewKernel {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}
```

### 2. Implement BatchKernel

```rust
use rustkernel_core::BatchKernel;

impl BatchKernel<MyInput, MyOutput> for MyNewKernel {
    async fn execute(&self, input: MyInput) -> Result<MyOutput> {
        // Implementation
    }
}
```

### 3. Add Input/Output Types

In `messages.rs`:

```rust
#[derive(Debug, Clone)]
pub struct MyInput {
    pub data: Vec<f64>,
    pub config: MyConfig,
}

#[derive(Debug, Clone)]
pub struct MyOutput {
    pub result: Vec<f64>,
    pub statistics: Stats,
}
```

### 4. Add Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_my_new_kernel_basic() {
        let kernel = MyNewKernel::new();
        let input = MyInput { /* ... */ };
        let result = kernel.execute(input).await.unwrap();
        assert!(/* expected conditions */);
    }

    #[test]
    fn test_my_new_kernel_metadata() {
        let kernel = MyNewKernel::new();
        assert_eq!(kernel.metadata().id, "domain/my-new-kernel");
    }
}
```

### 5. Register the Kernel

In the domain's `lib.rs`:

```rust
pub fn register_all(registry: &mut KernelRegistry) -> Result<()> {
    registry.register_batch(MyNewKernel::new())?;
    // ... other registrations
    Ok(())
}
```

### 6. Update Documentation

- Add kernel to domain's documentation page
- Update kernel count if needed
- Add to changelog

## Code Style

### Rust Style

- Follow standard Rust idioms
- Use `rustfmt` formatting
- Address all `clippy` warnings
- Document public APIs with `///` comments

### Naming Conventions

- Kernels: `PascalCase` (e.g., `PageRank`, `MonteCarloVaR`)
- Kernel IDs: `kebab-case` (e.g., `graph/page-rank`)
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`

### Error Handling

- Use `Result<T, KernelError>` for fallible operations
- Provide meaningful error messages
- Don't panic in library code

## Testing Guidelines

### Test Categories

1. **Unit tests**: Test individual functions
2. **Integration tests**: Test kernel execution
3. **Property tests**: For numerical algorithms (when applicable)

### Test Naming

```rust
#[test]
fn test_{function}_{scenario}_{expected_result}() {
    // e.g., test_pagerank_empty_graph_returns_empty()
}
```

### Coverage Goals

- New code should have >80% test coverage
- Critical paths should have 100% coverage

## Questions?

- Open a GitHub issue for bugs or feature requests
- Use discussions for questions
- Email maintainers for sensitive issues

Thank you for contributing!
