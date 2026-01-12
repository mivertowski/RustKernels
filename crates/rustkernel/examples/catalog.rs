//! Catalog Example
//!
//! Demonstrates browsing the kernel catalog.
//!
//! Run with: `cargo run --example catalog`

fn main() {
    println!("=== RustKernels Catalog ===\n");

    println!("Version: {}", rustkernel::version::VERSION);
    println!(
        "Min RingKernel: {}",
        rustkernel::version::MIN_RINGKERNEL_VERSION
    );
    println!();

    // List all domains
    let domains = rustkernel::catalog::domains();
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
        rustkernel::catalog::total_kernel_count()
    );
    println!();

    // Show enabled domains
    let enabled = rustkernel::catalog::enabled_domains();
    println!("Enabled domains: {:?}", enabled);
}
