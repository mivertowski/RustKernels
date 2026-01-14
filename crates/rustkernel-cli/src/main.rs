//! RustKernels CLI tool.
//!
//! Provides commands for kernel management, discovery, and validation.

use clap::{Parser, Subcommand};
use rustkernels::catalog::{DomainInfo, domains, enabled_domains, total_kernel_count};
use rustkernel_core::{domain::Domain, kernel::KernelMode, registry::KernelRegistry};
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
        /// Filter by domain (e.g., graph, ml, compliance)
        #[arg(short, long)]
        domain: Option<String>,

        /// Filter by mode (batch/ring)
        #[arg(short, long)]
        mode: Option<String>,
    },

    /// Show kernel registry statistics
    Stats,

    /// List available domains with details
    Domains {
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Show kernel information
    Info {
        /// Kernel ID (e.g., graph/pagerank)
        kernel_id: String,
    },

    /// Validate kernel configuration
    Validate {
        /// Kernel ID to validate
        kernel_id: String,
    },

    /// Check system compatibility
    Check {
        /// Check all backends
        #[arg(long)]
        all_backends: bool,
    },

    /// Show enabled features
    Features,

    /// Run a simple benchmark
    Bench {
        /// Domain to benchmark
        #[arg(short, long)]
        domain: Option<String>,

        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: u32,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    match cli.command {
        Commands::List { domain, mode } => {
            cmd_list(domain, mode)?;
        }

        Commands::Stats => {
            cmd_stats()?;
        }

        Commands::Domains { detailed } => {
            cmd_domains(detailed)?;
        }

        Commands::Info { kernel_id } => {
            cmd_info(&kernel_id)?;
        }

        Commands::Validate { kernel_id } => {
            cmd_validate(&kernel_id)?;
        }

        Commands::Check { all_backends } => {
            cmd_check(all_backends)?;
        }

        Commands::Features => {
            cmd_features()?;
        }

        Commands::Bench { domain, iterations } => {
            cmd_bench(domain, iterations)?;
        }
    }

    Ok(())
}

fn cmd_list(domain_filter: Option<String>, mode_filter: Option<String>) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              RustKernels Kernel Catalogue                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let all_domains = domains();
    let filter_domain = domain_filter.as_ref().and_then(|d| {
        all_domains
            .iter()
            .find(|info| info.feature == d.to_lowercase())
    });

    if let Some(filter) = &filter_domain {
        print_domain_kernels(filter, mode_filter.as_deref());
    } else if domain_filter.is_some() {
        println!("Unknown domain. Available domains:");
        for d in &all_domains {
            println!("  - {} (feature: {})", d.name, d.feature);
        }
    } else {
        // List all domains
        for info in &all_domains {
            print_domain_kernels(info, mode_filter.as_deref());
        }
    }

    println!(
        "\nTotal: {} kernels across {} domains",
        total_kernel_count(),
        all_domains.len()
    );

    Ok(())
}

fn print_domain_kernels(info: &DomainInfo, mode_filter: Option<&str>) {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ {} ({} kernels)", info.name, info.kernel_count);
    println!("│ {}", info.description);
    println!("├─────────────────────────────────────────────────────────────────┤");

    // Get kernel information for this domain
    let kernels = get_domain_kernels(&info.domain);

    for (kernel_id, kernel_mode, desc) in &kernels {
        let mode_str = match kernel_mode {
            KernelMode::Batch => "Batch",
            KernelMode::Ring => "Ring ",
        };

        // Apply mode filter if specified
        if let Some(filter) = mode_filter {
            let filter_lower = filter.to_lowercase();
            let matches = match kernel_mode {
                KernelMode::Batch => filter_lower == "batch",
                KernelMode::Ring => filter_lower == "ring",
            };
            if !matches {
                continue;
            }
        }

        println!("│  [{mode_str}] {kernel_id:<30} - {desc}");
    }

    println!("└─────────────────────────────────────────────────────────────────┘\n");
}

fn get_domain_kernels(domain: &Domain) -> Vec<(String, KernelMode, String)> {
    match domain {
        Domain::GraphAnalytics => vec![
            (
                "graph/degree_centrality".into(),
                KernelMode::Ring,
                "Node degree centrality".into(),
            ),
            (
                "graph/betweenness_centrality".into(),
                KernelMode::Ring,
                "Brandes algorithm".into(),
            ),
            (
                "graph/closeness_centrality".into(),
                KernelMode::Ring,
                "BFS-based centrality".into(),
            ),
            (
                "graph/eigenvector_centrality".into(),
                KernelMode::Ring,
                "Power iteration".into(),
            ),
            (
                "graph/pagerank".into(),
                KernelMode::Ring,
                "PageRank with teleport".into(),
            ),
            (
                "graph/katz_centrality".into(),
                KernelMode::Ring,
                "Attenuated paths".into(),
            ),
            (
                "graph/modularity".into(),
                KernelMode::Batch,
                "Community modularity Q".into(),
            ),
            (
                "graph/louvain".into(),
                KernelMode::Batch,
                "Louvain community detection".into(),
            ),
            (
                "graph/triangle_counting".into(),
                KernelMode::Ring,
                "Local triangle enumeration".into(),
            ),
            (
                "graph/motif_detection".into(),
                KernelMode::Batch,
                "K-node subgraph census".into(),
            ),
            (
                "graph/jaccard_similarity".into(),
                KernelMode::Batch,
                "Set similarity".into(),
            ),
            (
                "graph/cosine_similarity".into(),
                KernelMode::Batch,
                "Vector similarity".into(),
            ),
            (
                "graph/adamic_adar".into(),
                KernelMode::Batch,
                "Link prediction index".into(),
            ),
            (
                "graph/density".into(),
                KernelMode::Batch,
                "Graph density metric".into(),
            ),
            (
                "graph/clustering_coeff".into(),
                KernelMode::Batch,
                "Clustering coefficient".into(),
            ),
        ],
        Domain::StatisticalML => vec![
            (
                "ml/kmeans".into(),
                KernelMode::Batch,
                "K-Means++ clustering".into(),
            ),
            (
                "ml/dbscan".into(),
                KernelMode::Batch,
                "Density-based clustering".into(),
            ),
            (
                "ml/hierarchical".into(),
                KernelMode::Batch,
                "Agglomerative clustering".into(),
            ),
            (
                "ml/isolation_forest".into(),
                KernelMode::Batch,
                "Anomaly detection".into(),
            ),
            (
                "ml/lof".into(),
                KernelMode::Batch,
                "Local outlier factor".into(),
            ),
            (
                "ml/ensemble_voting".into(),
                KernelMode::Batch,
                "Weighted voting".into(),
            ),
        ],
        Domain::Compliance => vec![
            (
                "compliance/circular_flow".into(),
                KernelMode::Ring,
                "SCC detection".into(),
            ),
            (
                "compliance/reciprocity".into(),
                KernelMode::Ring,
                "Mutual transactions".into(),
            ),
            (
                "compliance/rapid_movement".into(),
                KernelMode::Ring,
                "Velocity analysis".into(),
            ),
            (
                "compliance/aml_patterns".into(),
                KernelMode::Ring,
                "Multi-pattern FSM".into(),
            ),
            (
                "compliance/kyc_scoring".into(),
                KernelMode::Batch,
                "Risk factor aggregation".into(),
            ),
            (
                "compliance/entity_resolution".into(),
                KernelMode::Batch,
                "Fuzzy matching".into(),
            ),
            (
                "compliance/sanctions".into(),
                KernelMode::Ring,
                "OFAC/UN/EU screening".into(),
            ),
            (
                "compliance/pep".into(),
                KernelMode::Ring,
                "PEP screening".into(),
            ),
            (
                "compliance/txmon".into(),
                KernelMode::Ring,
                "Real-time monitoring".into(),
            ),
        ],
        Domain::TemporalAnalysis => vec![
            (
                "temporal/arima".into(),
                KernelMode::Batch,
                "ARIMA forecasting".into(),
            ),
            (
                "temporal/prophet".into(),
                KernelMode::Batch,
                "Prophet decomposition".into(),
            ),
            (
                "temporal/changepoint".into(),
                KernelMode::Batch,
                "PELT detection".into(),
            ),
            (
                "temporal/anomaly".into(),
                KernelMode::Ring,
                "Statistical thresholds".into(),
            ),
            (
                "temporal/stl".into(),
                KernelMode::Batch,
                "STL decomposition".into(),
            ),
            (
                "temporal/trend".into(),
                KernelMode::Batch,
                "Trend extraction".into(),
            ),
            (
                "temporal/volatility".into(),
                KernelMode::Ring,
                "GARCH models".into(),
            ),
        ],
        Domain::RiskAnalytics => vec![
            (
                "risk/credit_scoring".into(),
                KernelMode::Ring,
                "PD/LGD/EAD calculation".into(),
            ),
            (
                "risk/monte_carlo_var".into(),
                KernelMode::Ring,
                "Monte Carlo VaR".into(),
            ),
            (
                "risk/portfolio_risk".into(),
                KernelMode::Ring,
                "Correlation-adjusted VaR".into(),
            ),
            (
                "risk/stress_testing".into(),
                KernelMode::Batch,
                "Scenario-based shocks".into(),
            ),
        ],
        Domain::Banking => vec![(
            "banking/fraud_pattern".into(),
            KernelMode::Ring,
            "Pattern matching".into(),
        )],
        Domain::BehavioralAnalytics => vec![
            (
                "behavioral/profiling".into(),
                KernelMode::Ring,
                "Feature extraction".into(),
            ),
            (
                "behavioral/anomaly_profiling".into(),
                KernelMode::Ring,
                "Deviation scoring".into(),
            ),
            (
                "behavioral/fraud_signature".into(),
                KernelMode::Ring,
                "Known patterns".into(),
            ),
            (
                "behavioral/causal_graph".into(),
                KernelMode::Batch,
                "DAG inference".into(),
            ),
            (
                "behavioral/forensic_query".into(),
                KernelMode::Batch,
                "Historical search".into(),
            ),
            (
                "behavioral/event_correlation".into(),
                KernelMode::Ring,
                "Temporal correlation".into(),
            ),
        ],
        Domain::OrderMatching => vec![(
            "orderbook/matching_engine".into(),
            KernelMode::Ring,
            "Price-time priority".into(),
        )],
        Domain::ProcessIntelligence => vec![
            (
                "procint/dfg".into(),
                KernelMode::Batch,
                "DFG construction".into(),
            ),
            (
                "procint/partial_order".into(),
                KernelMode::Batch,
                "Concurrency detection".into(),
            ),
            (
                "procint/conformance".into(),
                KernelMode::Ring,
                "Multi-model checking".into(),
            ),
            (
                "procint/ocpm".into(),
                KernelMode::Batch,
                "Object-centric PM".into(),
            ),
        ],
        Domain::Clearing => vec![
            (
                "clearing/validation".into(),
                KernelMode::Batch,
                "Trade validation".into(),
            ),
            (
                "clearing/dvp".into(),
                KernelMode::Ring,
                "Delivery vs payment".into(),
            ),
            (
                "clearing/netting".into(),
                KernelMode::Batch,
                "Multilateral netting".into(),
            ),
            (
                "clearing/settlement".into(),
                KernelMode::Ring,
                "Settlement execution".into(),
            ),
            (
                "clearing/zero_balance".into(),
                KernelMode::Batch,
                "Efficiency analysis".into(),
            ),
        ],
        Domain::TreasuryManagement => vec![
            (
                "treasury/cashflow".into(),
                KernelMode::Batch,
                "Cash flow forecasting".into(),
            ),
            (
                "treasury/collateral".into(),
                KernelMode::Batch,
                "Collateral optimization".into(),
            ),
            (
                "treasury/fx_hedging".into(),
                KernelMode::Batch,
                "FX exposure".into(),
            ),
            (
                "treasury/interest_rate".into(),
                KernelMode::Batch,
                "Duration/convexity".into(),
            ),
            (
                "treasury/liquidity".into(),
                KernelMode::Batch,
                "LCR/NSFR optimization".into(),
            ),
        ],
        Domain::Accounting => vec![
            (
                "accounting/coa_mapping".into(),
                KernelMode::Batch,
                "Chart of accounts".into(),
            ),
            (
                "accounting/journal".into(),
                KernelMode::Batch,
                "GL mapping".into(),
            ),
            (
                "accounting/reconciliation".into(),
                KernelMode::Batch,
                "Account matching".into(),
            ),
            (
                "accounting/network".into(),
                KernelMode::Batch,
                "Intercompany analysis".into(),
            ),
            (
                "accounting/temporal_corr".into(),
                KernelMode::Batch,
                "Account correlations".into(),
            ),
        ],
        Domain::PaymentProcessing => vec![
            (
                "payments/processing".into(),
                KernelMode::Ring,
                "Transaction execution".into(),
            ),
            (
                "payments/flow_analysis".into(),
                KernelMode::Batch,
                "Payment flow metrics".into(),
            ),
        ],
        Domain::FinancialAudit => vec![
            (
                "audit/feature_extraction".into(),
                KernelMode::Batch,
                "Audit feature vectors".into(),
            ),
            (
                "audit/hypergraph".into(),
                KernelMode::Batch,
                "Multi-way relationships".into(),
            ),
        ],
        Domain::Core => vec![
            (
                "core/vector_add".into(),
                KernelMode::Batch,
                "Simple validation".into(),
            ),
            (
                "core/echo".into(),
                KernelMode::Ring,
                "Message round-trip".into(),
            ),
        ],
        // Wildcard for future domains (Domain is non-exhaustive)
        _ => vec![],
    }
}

fn cmd_stats() -> anyhow::Result<()> {
    let registry = KernelRegistry::new();
    let stats = registry.stats();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              RustKernels Registry Statistics                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Total kernels:  {}", stats.total);
    println!("Batch kernels:  {}", stats.batch_kernels);
    println!("Ring kernels:   {}", stats.ring_kernels);
    println!();

    println!("Kernels by domain:");
    println!("──────────────────────────────────────────────────────────────────");

    let domain_infos = domains();
    for info in &domain_infos {
        let bar_len = info.kernel_count.min(30);
        let bar: String = "█".repeat(bar_len);
        println!("  {:<25} {:>3} {}", info.name, info.kernel_count, bar);
    }

    println!("──────────────────────────────────────────────────────────────────");
    println!("  {:<25} {:>3}", "TOTAL", total_kernel_count());

    Ok(())
}

fn cmd_domains(detailed: bool) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     Available Domains                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let all_domains = domains();

    // Group by priority
    let p1: Vec<_> = all_domains
        .iter()
        .filter(|d| {
            matches!(
                d.domain,
                Domain::GraphAnalytics
                    | Domain::StatisticalML
                    | Domain::Compliance
                    | Domain::TemporalAnalysis
                    | Domain::RiskAnalytics
            )
        })
        .collect();

    let p2: Vec<_> = all_domains
        .iter()
        .filter(|d| {
            matches!(
                d.domain,
                Domain::Banking
                    | Domain::BehavioralAnalytics
                    | Domain::OrderMatching
                    | Domain::ProcessIntelligence
                    | Domain::Clearing
            )
        })
        .collect();

    let p3: Vec<_> = all_domains
        .iter()
        .filter(|d| {
            matches!(
                d.domain,
                Domain::TreasuryManagement
                    | Domain::Accounting
                    | Domain::PaymentProcessing
                    | Domain::FinancialAudit
            )
        })
        .collect();

    println!("Priority 1 (High Value):");
    println!("────────────────────────");
    for d in &p1 {
        print_domain_info(d, detailed);
    }

    println!("\nPriority 2 (Medium):");
    println!("────────────────────");
    for d in &p2 {
        print_domain_info(d, detailed);
    }

    println!("\nPriority 3 (Lower):");
    println!("───────────────────");
    for d in &p3 {
        print_domain_info(d, detailed);
    }

    Ok(())
}

fn print_domain_info(info: &DomainInfo, detailed: bool) {
    if detailed {
        println!("  {} (feature: {})", info.name, info.feature);
        println!("    Description: {}", info.description);
        println!("    Kernels: {}", info.kernel_count);
        println!();
    } else {
        println!(
            "  {:<25} {:>2} kernels  (--features {})",
            info.name, info.kernel_count, info.feature
        );
    }
}

fn cmd_info(kernel_id: &str) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     Kernel Information                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Parse kernel ID to find domain
    let parts: Vec<&str> = kernel_id.split('/').collect();
    if parts.len() != 2 {
        println!("Invalid kernel ID format. Use: domain/kernel_name");
        println!("Example: graph/pagerank");
        return Ok(());
    }

    let domain_str = parts[0];
    let kernel_name = parts[1];

    let all_domains = domains();
    let domain_info = all_domains.iter().find(|d| d.feature == domain_str);

    if let Some(info) = domain_info {
        let kernels = get_domain_kernels(&info.domain);
        if let Some((id, mode, desc)) = kernels.iter().find(|(id, _, _)| id == kernel_id) {
            println!("Kernel ID:    {}", id);
            println!("Domain:       {} ({:?})", info.name, info.domain);
            println!("Mode:         {:?}", mode);
            println!("Description:  {}", desc);
            println!();

            println!("Usage:");
            match mode {
                KernelMode::Batch => {
                    println!("  // Batch kernel - CPU orchestrated");
                    println!("  let kernel = {}::new();", to_struct_name(kernel_name));
                    println!("  let result = kernel.execute(&input).await?;");
                }
                KernelMode::Ring => {
                    println!("  // Ring kernel - GPU persistent actor");
                    println!("  let kernel = runtime.launch(\"{}\", options).await?;", id);
                    println!("  kernel.send(request).await?;");
                    println!("  let response = kernel.receive().await?;");
                }
            }
        } else {
            println!(
                "Kernel '{}' not found in domain '{}'",
                kernel_name, domain_str
            );
        }
    } else {
        println!("Unknown domain: {}", domain_str);
        println!("\nAvailable domains:");
        for d in &all_domains {
            println!("  - {}", d.feature);
        }
    }

    Ok(())
}

fn to_struct_name(kernel_name: &str) -> String {
    kernel_name
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

fn cmd_validate(kernel_id: &str) -> anyhow::Result<()> {
    println!("Validating kernel: {}", kernel_id);
    println!();

    // Parse and validate kernel ID format
    let parts: Vec<&str> = kernel_id.split('/').collect();
    if parts.len() != 2 {
        println!("✗ Invalid kernel ID format");
        return Ok(());
    }

    println!("✓ Kernel ID format valid");

    let domain_str = parts[0];
    let all_domains = domains();

    if all_domains.iter().any(|d| d.feature == domain_str) {
        println!("✓ Domain '{}' exists", domain_str);
    } else {
        println!("✗ Unknown domain: {}", domain_str);
        return Ok(());
    }

    // Check if feature is enabled
    let enabled = enabled_domains();
    if enabled.contains(&domain_str) {
        println!("✓ Domain feature enabled");
    } else {
        println!(
            "⚠ Domain feature not enabled (add --features {})",
            domain_str
        );
    }

    println!();
    println!("Validation complete!");

    Ok(())
}

fn cmd_check(all_backends: bool) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                  System Compatibility Check                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("RustKernels Version: {}", rustkernels::version::VERSION);
    println!(
        "Min RingKernel:      {}",
        rustkernels::version::MIN_RINGKERNEL_VERSION
    );
    println!();

    println!("Enabled Features:");
    let enabled = enabled_domains();
    for feature in &enabled {
        println!("  ✓ {}", feature);
    }
    println!();

    if all_backends {
        println!("Backend Status:");
        println!("  CPU Backend:    ✓ Always available");
        println!("  CUDA Backend:   ? Requires NVIDIA GPU with CUDA");
        println!("  WebGPU Backend: ? Requires compatible GPU");
        println!();
        println!("To check GPU availability, use the ringkernel CLI.");
    } else {
        println!("Backend Status: CPU ✓");
        println!("  (use --all-backends for detailed check)");
    }

    Ok(())
}

fn cmd_features() -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     Enabled Features                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let all_domains = domains();
    let enabled = enabled_domains();

    println!("Feature Status:");
    println!("──────────────────────────────────────────────────────────────────");

    for info in &all_domains {
        let status = if enabled.contains(&info.feature) {
            "✓"
        } else {
            "✗"
        };
        let kernels = if enabled.contains(&info.feature) {
            format!("{} kernels", info.kernel_count)
        } else {
            "disabled".to_string()
        };
        println!("  {} {:<15} {}", status, info.feature, kernels);
    }

    println!("──────────────────────────────────────────────────────────────────");

    let enabled_count: usize = all_domains
        .iter()
        .filter(|d| enabled.contains(&d.feature))
        .map(|d| d.kernel_count)
        .sum();

    println!(
        "  {} of {} kernels enabled",
        enabled_count,
        total_kernel_count()
    );
    println!();

    println!("To enable all features:");
    println!("  cargo build --features full");

    Ok(())
}

fn cmd_bench(domain: Option<String>, iterations: u32) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     Simple Benchmark                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    if let Some(d) = &domain {
        println!("Domain: {}", d);
    } else {
        println!("Domain: all");
    }
    println!("Iterations: {}", iterations);
    println!();

    println!("For comprehensive benchmarks, use:");
    println!("  cargo bench --package rustkernel");
    println!();
    println!("Or for specific domain:");
    println!("  cargo bench --package rustkernel-graph");

    Ok(())
}
