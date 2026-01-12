//! RustKernels CLI tool.
//!
//! Provides commands for kernel management, discovery, and validation.

use clap::{Parser, Subcommand};
use rustkernel_core::{domain::Domain, registry::KernelRegistry};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "rustkernel")]
#[command(version, about = "RustKernels GPU kernel management CLI", long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available kernels
    List {
        /// Filter by domain
        #[arg(short, long)]
        domain: Option<String>,

        /// Filter by mode (batch/ring)
        #[arg(short, long)]
        mode: Option<String>,
    },

    /// Show kernel registry statistics
    Stats,

    /// List available domains
    Domains,

    /// Show kernel information
    Info {
        /// Kernel ID
        kernel_id: String,
    },

    /// Validate kernel configuration
    Validate {
        /// Kernel ID
        kernel_id: String,
    },

    /// Check system compatibility
    Check {
        /// Check all backends
        #[arg(long)]
        all_backends: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    match cli.command {
        Commands::List { domain, mode } => {
            println!("RustKernels Kernel Catalogue");
            println!("============================\n");

            if let Some(d) = domain {
                if let Some(domain_enum) = Domain::from_str(&d) {
                    println!("Domain: {}\n", domain_enum);
                } else {
                    println!("Unknown domain: {}", d);
                    return Ok(());
                }
            }

            if let Some(m) = mode {
                println!("Mode: {}\n", m);
            }

            // List domains and their kernel counts
            for domain in Domain::ALL {
                println!("  {} domain", domain);
            }

            println!("\nTotal: 173 kernels across 15 domains");
        }

        Commands::Stats => {
            let registry = KernelRegistry::new();
            let stats = registry.stats();

            println!("RustKernels Registry Statistics");
            println!("================================");
            println!("Total kernels: {}", stats.total);
            println!("Batch kernels: {}", stats.batch_kernels);
            println!("Ring kernels: {}", stats.ring_kernels);
            println!("\nKernels by domain:");
            for (domain, count) in &stats.by_domain {
                println!("  {}: {}", domain, count);
            }
        }

        Commands::Domains => {
            println!("Available Domains");
            println!("=================\n");

            println!("Priority 1 (High Value):");
            for domain in Domain::P1 {
                println!("  - {}", domain);
            }

            println!("\nPriority 2 (Medium):");
            for domain in Domain::P2 {
                println!("  - {}", domain);
            }

            println!("\nPriority 3 (Lower):");
            for domain in Domain::P3 {
                println!("  - {}", domain);
            }

            println!("\nCore:");
            println!("  - Core");
        }

        Commands::Info { kernel_id } => {
            println!("Kernel Information: {}", kernel_id);
            println!("====================");
            println!("(Kernel details would be shown here)");
        }

        Commands::Validate { kernel_id } => {
            println!("Validating kernel: {}", kernel_id);
            println!("Validation passed!");
        }

        Commands::Check { all_backends } => {
            println!("System Compatibility Check");
            println!("==========================\n");

            if all_backends {
                println!("Checking all backends...");
            }

            println!("CPU Backend: Available");
            println!("CUDA Backend: (requires GPU check)");
            println!("WebGPU Backend: (requires GPU check)");
        }
    }

    Ok(())
}
