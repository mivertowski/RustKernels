//! Licensing Example
//!
//! Demonstrates the licensing system.
//!
//! Run with: `cargo run --example licensing`

use rustkernel::core::domain::Domain;
use rustkernel::core::license::{
    DevelopmentLicense, License, LicenseGuard, LicenseTier, LicenseValidator,
    StandardLicenseValidator,
};

fn main() {
    println!("=== RustKernels Licensing Demo ===\n");

    // Development License
    println!("1. Development License");
    println!("   -------------------");
    let dev = DevelopmentLicense;
    demonstrate_license(&dev, "Development");
    println!();

    // Community License
    println!("2. Community License");
    println!("   ------------------");
    let community = License::community("Demo User");
    let community_validator = StandardLicenseValidator::new(community);
    demonstrate_license(&community_validator, "Community");
    println!();

    // Professional License
    println!("3. Professional License");
    println!("   ---------------------");
    let mut pro_domains = std::collections::HashSet::new();
    pro_domains.insert(Domain::GraphAnalytics);
    pro_domains.insert(Domain::StatisticalML);
    pro_domains.insert(Domain::Compliance);
    pro_domains.insert(Domain::RiskAnalytics);
    let professional = License::professional("Demo Corp", pro_domains, None);
    let pro_validator = StandardLicenseValidator::new(professional);
    demonstrate_license(&pro_validator, "Professional");
    println!();

    // Enterprise License
    println!("4. Enterprise License");
    println!("   -------------------");
    let enterprise = License::enterprise("Enterprise Corp", None);
    let enterprise_validator = StandardLicenseValidator::new(enterprise);
    demonstrate_license(&enterprise_validator, "Enterprise");
    println!();

    // License Guard Example
    println!("5. License Guard Usage");
    println!("   --------------------");
    let guard = LicenseGuard::new(&dev, Domain::GraphAnalytics);
    println!("   Guard for GraphAnalytics domain:");
    println!("   - Domain check: {:?}", guard.check());
    println!(
        "   - Feature check: {:?}",
        guard.check_feature("GraphAnalytics.PageRank")
    );
    println!("   - GPU native check: {:?}", guard.check_gpu_native());
}

fn demonstrate_license<V: LicenseValidator + ?Sized>(validator: &V, name: &str) {
    println!("   Tier: {:?}", validator.tier());
    println!("   GPU Native: {}", validator.gpu_native_enabled());
    println!("   Max Kernels: {:?}", validator.max_kernels());
    println!("   Expires: {:?}", validator.expires_at());
    println!();

    // Test some domains
    let test_domains = [
        Domain::GraphAnalytics,
        Domain::StatisticalML,
        Domain::RiskAnalytics,
        Domain::Banking,
    ];

    println!("   Domain Access:");
    for domain in &test_domains {
        let result = validator.validate_domain(*domain);
        let status = if result.is_ok() { "allowed" } else { "denied " };
        println!("   - {:?}: {}", domain, status);
    }
}
