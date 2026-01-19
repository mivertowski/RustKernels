//! Catalog Example
//!
//! Demonstrates browsing the kernel catalog.
//!
//! Run with: `cargo run --example catalog`

fn main() {
    println!("=== RustKernels Catalog ===\n");

    println!("Version: {}", rustkernels::version::VERSION);
    println!(
        "Min RingKernel: {}",
        rustkernels::version::MIN_RINGKERNEL_VERSION
    );
    println!();

    // List all domains
    let domains = rustkernels::catalog::domains();
    println!("Available Domains ({}):", domains.len());
    println!("------------------------");

    for domain in &domains {
        println!("  {} (feature: {})", domain.name, domain.feature);
        println!(
            "    {} kernels - {}",
            domain.kernel_count, domain.description
        );
        println!();
    }

    println!(
        "Total kernel count: {}",
        rustkernels::catalog::total_kernel_count()
    );
    println!();

    // Show enabled domains
    let enabled = rustkernels::catalog::enabled_domains();
    println!("Enabled domains: {:?}", enabled);
}
