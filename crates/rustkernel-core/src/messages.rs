//! Message types and traits for Ring kernel communication.
//!
//! This module provides the base message infrastructure for GPU-native
//! persistent actor communication using RingKernel's K2K messaging.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

// Re-export core message types from ringkernel
pub use ringkernel_core::{MessageHeader, MessageId, RingMessage};

/// Global message ID counter for correlation tracking.
static MESSAGE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique message ID.
#[must_use]
pub fn next_message_id() -> MessageId {
    MessageId(MESSAGE_COUNTER.fetch_add(1, Ordering::SeqCst))
}

/// Correlation ID for request-response pairing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorrelationId(pub u64);

impl CorrelationId {
    /// Create a new correlation ID.
    #[must_use]
    pub fn new() -> Self {
        Self(MESSAGE_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Create from a raw value.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

impl Default for CorrelationId {
    fn default() -> Self {
        Self::new()
    }
}

/// Base trait for kernel request messages.
pub trait KernelRequest: RingMessage + Send + Sync {
    /// Get the correlation ID for this request.
    fn correlation_id(&self) -> CorrelationId;

    /// Set the correlation ID.
    fn set_correlation_id(&mut self, id: CorrelationId);
}

/// Base trait for kernel response messages.
pub trait KernelResponse: RingMessage + Send + Sync {
    /// Get the correlation ID that this response corresponds to.
    fn correlation_id(&self) -> CorrelationId;

    /// Check if the response indicates success.
    fn is_success(&self) -> bool;

    /// Get any error message if the response indicates failure.
    fn error_message(&self) -> Option<&str>;
}

/// Generic result wrapper for kernel responses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelResult<T> {
    /// Correlation ID linking to the original request.
    pub correlation_id: CorrelationId,
    /// The result data, if successful.
    pub data: Option<T>,
    /// Error message, if failed.
    pub error: Option<String>,
}

impl<T> KernelResult<T> {
    /// Create a successful result.
    pub fn success(correlation_id: CorrelationId, data: T) -> Self {
        Self {
            correlation_id,
            data: Some(data),
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(correlation_id: CorrelationId, error: impl Into<String>) -> Self {
        Self {
            correlation_id,
            data: None,
            error: Some(error.into()),
        }
    }

    /// Check if this is a successful result.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.data.is_some() && self.error.is_none()
    }

    /// Convert to a standard Result type.
    pub fn into_result(self) -> Result<T, String> {
        match (self.data, self.error) {
            (Some(data), None) => Ok(data),
            (_, Some(err)) => Err(err),
            (None, None) => Err("No data or error provided".to_string()),
        }
    }
}

/// Message type IDs for each domain.
///
/// These are used for routing and serialization in K2K messaging.
/// Each domain has a reserved range of IDs.
#[allow(missing_docs)]
pub mod type_ids {
    // Graph Analytics domain (100-199)
    pub const PAGERANK_REQUEST: u32 = 100;
    pub const PAGERANK_RESPONSE: u32 = 101;
    pub const DEGREE_CENTRALITY_REQUEST: u32 = 102;
    pub const DEGREE_CENTRALITY_RESPONSE: u32 = 103;
    pub const BETWEENNESS_REQUEST: u32 = 104;
    pub const BETWEENNESS_RESPONSE: u32 = 105;
    pub const CLOSENESS_REQUEST: u32 = 106;
    pub const CLOSENESS_RESPONSE: u32 = 107;
    pub const EIGENVECTOR_REQUEST: u32 = 108;
    pub const EIGENVECTOR_RESPONSE: u32 = 109;
    pub const KATZ_REQUEST: u32 = 110;
    pub const KATZ_RESPONSE: u32 = 111;
    pub const COMMUNITY_REQUEST: u32 = 120;
    pub const COMMUNITY_RESPONSE: u32 = 121;
    pub const MOTIF_REQUEST: u32 = 130;
    pub const MOTIF_RESPONSE: u32 = 131;
    pub const SIMILARITY_REQUEST: u32 = 140;
    pub const SIMILARITY_RESPONSE: u32 = 141;
    pub const METRICS_REQUEST: u32 = 150;
    pub const METRICS_RESPONSE: u32 = 151;

    // ML domain (200-299)
    pub const KMEANS_REQUEST: u32 = 200;
    pub const KMEANS_RESPONSE: u32 = 201;
    pub const DBSCAN_REQUEST: u32 = 202;
    pub const DBSCAN_RESPONSE: u32 = 203;
    pub const HIERARCHICAL_REQUEST: u32 = 204;
    pub const HIERARCHICAL_RESPONSE: u32 = 205;
    pub const ISOLATION_FOREST_REQUEST: u32 = 210;
    pub const ISOLATION_FOREST_RESPONSE: u32 = 211;
    pub const LOF_REQUEST: u32 = 212;
    pub const LOF_RESPONSE: u32 = 213;
    pub const REGRESSION_REQUEST: u32 = 220;
    pub const REGRESSION_RESPONSE: u32 = 221;

    // Compliance domain (300-399)
    pub const CIRCULAR_FLOW_REQUEST: u32 = 300;
    pub const CIRCULAR_FLOW_RESPONSE: u32 = 301;
    pub const RECIPROCITY_REQUEST: u32 = 302;
    pub const RECIPROCITY_RESPONSE: u32 = 303;
    pub const RAPID_MOVEMENT_REQUEST: u32 = 304;
    pub const RAPID_MOVEMENT_RESPONSE: u32 = 305;
    pub const AML_PATTERN_REQUEST: u32 = 306;
    pub const AML_PATTERN_RESPONSE: u32 = 307;
    pub const SANCTIONS_REQUEST: u32 = 310;
    pub const SANCTIONS_RESPONSE: u32 = 311;
    pub const KYC_REQUEST: u32 = 320;
    pub const KYC_RESPONSE: u32 = 321;
    pub const TRANSACTION_MONITOR_REQUEST: u32 = 330;
    pub const TRANSACTION_MONITOR_RESPONSE: u32 = 331;

    // Risk domain (400-499)
    pub const CREDIT_RISK_REQUEST: u32 = 400;
    pub const CREDIT_RISK_RESPONSE: u32 = 401;
    pub const MONTE_CARLO_VAR_REQUEST: u32 = 410;
    pub const MONTE_CARLO_VAR_RESPONSE: u32 = 411;
    pub const PORTFOLIO_RISK_REQUEST: u32 = 412;
    pub const PORTFOLIO_RISK_RESPONSE: u32 = 413;
    pub const STRESS_TEST_REQUEST: u32 = 420;
    pub const STRESS_TEST_RESPONSE: u32 = 421;

    // Temporal domain (500-599)
    pub const ARIMA_REQUEST: u32 = 500;
    pub const ARIMA_RESPONSE: u32 = 501;
    pub const PROPHET_REQUEST: u32 = 502;
    pub const PROPHET_RESPONSE: u32 = 503;
    pub const CHANGE_POINT_REQUEST: u32 = 510;
    pub const CHANGE_POINT_RESPONSE: u32 = 511;
    pub const VOLATILITY_REQUEST: u32 = 520;
    pub const VOLATILITY_RESPONSE: u32 = 521;

    // OrderBook domain (600-699)
    pub const ORDER_SUBMIT_REQUEST: u32 = 600;
    pub const ORDER_SUBMIT_RESPONSE: u32 = 601;
    pub const ORDER_CANCEL_REQUEST: u32 = 602;
    pub const ORDER_CANCEL_RESPONSE: u32 = 603;
    pub const ORDER_MODIFY_REQUEST: u32 = 604;
    pub const ORDER_MODIFY_RESPONSE: u32 = 605;
    pub const BOOK_QUERY_REQUEST: u32 = 610;
    pub const BOOK_QUERY_RESPONSE: u32 = 611;

    // Clearing domain (700-799)
    pub const CLEARING_VALIDATION_REQUEST: u32 = 700;
    pub const CLEARING_VALIDATION_RESPONSE: u32 = 701;
    pub const DVP_MATCHING_REQUEST: u32 = 710;
    pub const DVP_MATCHING_RESPONSE: u32 = 711;
    pub const NETTING_REQUEST: u32 = 720;
    pub const NETTING_RESPONSE: u32 = 721;
    pub const SETTLEMENT_REQUEST: u32 = 730;
    pub const SETTLEMENT_RESPONSE: u32 = 731;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_id() {
        let id1 = CorrelationId::new();
        let id2 = CorrelationId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_message_id_generation() {
        let id1 = next_message_id();
        let id2 = next_message_id();
        assert!(id2.0 > id1.0);
    }

    #[test]
    fn test_kernel_result_success() {
        let result = KernelResult::success(CorrelationId::new(), 42);
        assert!(result.is_success());
        assert_eq!(result.into_result(), Ok(42));
    }

    #[test]
    fn test_kernel_result_failure() {
        let result: KernelResult<i32> = KernelResult::failure(CorrelationId::new(), "error");
        assert!(!result.is_success());
        assert_eq!(result.into_result(), Err("error".to_string()));
    }
}
