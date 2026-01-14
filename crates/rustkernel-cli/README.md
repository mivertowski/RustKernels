# rustkernel-cli

[![Crates.io](https://img.shields.io/crates/v/rustkernel-cli.svg)](https://crates.io/crates/rustkernel-cli)
[![Documentation](https://docs.rs/rustkernel-cli/badge.svg)](https://docs.rs/rustkernel-cli)
[![License](https://img.shields.io/crates/l/rustkernel-cli.svg)](LICENSE)

CLI tool for RustKernels management.

## Features

- List available kernels
- Query kernel metadata
- Validate kernel configurations
- License management

## Installation

```bash
cargo install rustkernel-cli
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
rustkernel-cli = "0.1.0"
```

## Usage

```bash
# List all kernels
rustkernel-cli list

# Show kernel info
rustkernel-cli info graph/pagerank

# List kernels by domain
rustkernel-cli list --domain GraphAnalytics

# Validate configuration
rustkernel-cli validate config.toml
```

## Commands

| Command | Description |
|---------|-------------|
| `list` | List available kernels |
| `info <kernel>` | Show kernel metadata |
| `validate <config>` | Validate configuration |
| `license` | License management |

## License

Apache-2.0
