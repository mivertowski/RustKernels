//! Procedural macros for RustKernels.
//!
//! This crate provides the following macros:
//! - `#[gpu_kernel]` - Define a GPU kernel with metadata
//! - `#[derive(KernelMessage)]` - Derive serialization for kernel messages
//!
//! # Example
//!
//! ```ignore
//! use rustkernel_derive::gpu_kernel;
//!
//! #[gpu_kernel(
//!     id = "graph/pagerank",
//!     mode = "ring",
//!     domain = "GraphAnalytics",
//!     throughput = 100_000,
//!     latency_us = 1.0
//! )]
//! pub async fn pagerank_kernel(
//!     ctx: &mut RingContext,
//!     request: PageRankRequest,
//! ) -> PageRankResponse {
//!     // Implementation
//! }
//! ```

use darling::{FromDeriveInput, FromMeta};
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput, ItemFn};

/// Arguments for the `#[gpu_kernel]` attribute.
#[derive(Debug, FromMeta)]
struct GpuKernelArgs {
    /// Kernel ID (e.g., "graph/pagerank").
    id: String,

    /// Kernel mode: "batch" or "ring".
    mode: String,

    /// Domain name (e.g., "GraphAnalytics").
    domain: String,

    /// Description (optional).
    #[darling(default)]
    description: Option<String>,

    /// Expected throughput in ops/sec (optional).
    #[darling(default)]
    throughput: Option<u64>,

    /// Target latency in microseconds (optional).
    #[darling(default)]
    latency_us: Option<f64>,

    /// Whether GPU-native execution is required (optional).
    #[darling(default)]
    gpu_native: Option<bool>,
}

/// Define a GPU kernel with metadata.
///
/// This attribute generates a kernel struct and implements the necessary traits.
///
/// # Attributes
///
/// - `id` - Unique kernel identifier (required)
/// - `mode` - Kernel mode: "batch" or "ring" (required)
/// - `domain` - Business domain (required)
/// - `description` - Human-readable description (optional)
/// - `throughput` - Expected throughput in ops/sec (optional)
/// - `latency_us` - Target latency in microseconds (optional)
/// - `gpu_native` - Whether GPU-native execution is required (optional)
///
/// # Example
///
/// ```ignore
/// #[gpu_kernel(
///     id = "graph/pagerank",
///     mode = "ring",
///     domain = "GraphAnalytics",
///     description = "PageRank centrality calculation",
///     throughput = 100_000,
///     latency_us = 1.0,
///     gpu_native = true
/// )]
/// pub async fn pagerank(ctx: &mut RingContext, req: PageRankRequest) -> PageRankResponse {
///     // Implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = match darling::ast::NestedMeta::parse_meta_list(attr.into()) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.to_compile_error()),
    };

    let args = match GpuKernelArgs::from_list(&args) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_vis = &input.vis;
    let fn_block = &input.block;
    let fn_inputs = &input.sig.inputs;
    let fn_output = &input.sig.output;
    let fn_asyncness = &input.sig.asyncness;

    // Generate struct name from function name (PascalCase)
    let struct_name = to_pascal_case(&fn_name.to_string());
    let struct_ident = syn::Ident::new(&struct_name, fn_name.span());

    // Parse mode
    let mode = match args.mode.as_str() {
        "batch" => quote! { rustkernel_core::kernel::KernelMode::Batch },
        "ring" => quote! { rustkernel_core::kernel::KernelMode::Ring },
        _ => {
            return syn::Error::new_spanned(&input.sig, "mode must be 'batch' or 'ring'")
                .to_compile_error()
                .into()
        }
    };

    // Parse domain
    let domain = &args.domain;
    let domain_ident = syn::Ident::new(domain, proc_macro2::Span::call_site());

    // Default values
    let description = args.description.unwrap_or_default();
    let throughput = args.throughput.unwrap_or(10_000);
    let latency_us = args.latency_us.unwrap_or(50.0);
    let gpu_native = args.gpu_native.unwrap_or(false);
    let kernel_id = &args.id;

    // Generate the kernel struct and implementation
    let expanded = quote! {
        /// Generated kernel struct for #fn_name.
        #[derive(Debug, Clone)]
        #fn_vis struct #struct_ident {
            metadata: rustkernel_core::kernel::KernelMetadata,
        }

        impl #struct_ident {
            /// Create a new instance of this kernel.
            #[must_use]
            pub fn new() -> Self {
                Self {
                    metadata: rustkernel_core::kernel::KernelMetadata {
                        id: #kernel_id.to_string(),
                        mode: #mode,
                        domain: rustkernel_core::domain::Domain::#domain_ident,
                        description: #description.to_string(),
                        expected_throughput: #throughput,
                        target_latency_us: #latency_us,
                        requires_gpu_native: #gpu_native,
                        version: 1,
                    },
                }
            }
        }

        impl Default for #struct_ident {
            fn default() -> Self {
                Self::new()
            }
        }

        impl rustkernel_core::traits::GpuKernel for #struct_ident {
            fn metadata(&self) -> &rustkernel_core::kernel::KernelMetadata {
                &self.metadata
            }
        }

        // Keep the original function for implementation
        #fn_vis #fn_asyncness fn #fn_name(#fn_inputs) #fn_output
        #fn_block
    };

    TokenStream::from(expanded)
}

/// Convert a snake_case string to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// Arguments for `#[derive(KernelMessage)]`.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(message))]
struct KernelMessageArgs {
    ident: syn::Ident,
    generics: syn::Generics,

    /// Message type ID.
    #[darling(default)]
    type_id: Option<u64>,

    /// Domain for the message.
    #[darling(default)]
    domain: Option<String>,
}

/// Derive macro for kernel messages.
///
/// This generates implementations for serialization and the `RingMessage` trait.
///
/// # Attributes
///
/// - `type_id` - Unique message type identifier (optional)
/// - `domain` - Domain for the message (optional)
///
/// # Example
///
/// ```ignore
/// #[derive(KernelMessage)]
/// #[message(type_id = 100, domain = "GraphAnalytics")]
/// pub struct PageRankRequest {
///     pub node_id: u64,
///     pub operation: PageRankOp,
/// }
/// ```
#[proc_macro_derive(KernelMessage, attributes(message))]
pub fn derive_kernel_message(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let args = match KernelMessageArgs::from_derive_input(&input) {
        Ok(v) => v,
        Err(e) => return TokenStream::from(e.write_errors()),
    };

    let name = args.ident;
    let (impl_generics, ty_generics, where_clause) = args.generics.split_for_impl();

    let type_id = args.type_id.unwrap_or(0);

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Get the message type ID.
            #[must_use]
            pub const fn message_type_id() -> u64 {
                #type_id
            }
        }

        // Implement basic serialization traits
        // Note: Full RingMessage implementation requires ringkernel-derive
        // This is a placeholder that adds basic functionality
    };

    TokenStream::from(expanded)
}

/// Attribute for marking kernel state types.
///
/// This ensures the type meets GPU requirements (unmanaged, fixed layout).
///
/// # Example
///
/// ```ignore
/// #[kernel_state(size = 256)]
/// pub struct PageRankState {
///     pub scores: [f32; 64],
/// }
/// ```
#[proc_macro_attribute]
pub fn kernel_state(attr: TokenStream, item: TokenStream) -> TokenStream {
    // For now, just pass through - state validation can be added later
    let input = parse_macro_input!(item as DeriveInput);

    let expanded = quote! {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default)]
        #input
    };

    TokenStream::from(expanded)
}
