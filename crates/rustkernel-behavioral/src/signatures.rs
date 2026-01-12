//! Fraud signature detection kernels.
//!
//! This module provides pattern-based fraud detection using
//! predefined fraud signatures.

use crate::types::{
    EventValue, FraudSignature, SignatureMatch, SignaturePattern, UserEvent,
};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::HashMap;

// ============================================================================
// Fraud Signature Detection Kernel
// ============================================================================

/// Fraud signature detection kernel.
///
/// Matches user events against known fraud patterns/signatures.
#[derive(Debug, Clone)]
pub struct FraudSignatureDetection {
    metadata: KernelMetadata,
}

impl Default for FraudSignatureDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl FraudSignatureDetection {
    /// Create a new fraud signature detection kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring(
                "behavioral/fraud-signatures",
                Domain::BehavioralAnalytics,
            )
            .with_description("Known fraud pattern signature matching")
            .with_throughput(150_000)
            .with_latency_us(30.0),
        }
    }

    /// Match events against fraud signatures.
    ///
    /// # Arguments
    /// * `events` - Events to analyze
    /// * `signatures` - Active fraud signatures to match against
    pub fn compute(events: &[UserEvent], signatures: &[FraudSignature]) -> Vec<SignatureMatch> {
        let mut matches = Vec::new();

        for signature in signatures.iter().filter(|s| s.active) {
            if let Some(match_result) = Self::match_signature(events, signature) {
                matches.push(match_result);
            }
        }

        // Sort by score descending
        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        matches
    }

    /// Match a single signature against events.
    fn match_signature(events: &[UserEvent], signature: &FraudSignature) -> Option<SignatureMatch> {
        match &signature.pattern {
            SignaturePattern::EventSequence(sequence) => {
                Self::match_event_sequence(events, signature, sequence)
            }
            SignaturePattern::EventAttributes(event_type, attrs) => {
                Self::match_event_attributes(events, signature, event_type, attrs)
            }
            SignaturePattern::TimeWindow {
                events: event_types,
                window_secs,
            } => Self::match_time_window(events, signature, event_types, *window_secs),
            SignaturePattern::CountThreshold {
                event_type,
                count,
                window_secs,
            } => Self::match_count_threshold(events, signature, event_type, *count, *window_secs),
            SignaturePattern::Regex(pattern) => Self::match_regex(events, signature, pattern),
        }
    }

    /// Match an event sequence pattern.
    fn match_event_sequence(
        events: &[UserEvent],
        signature: &FraudSignature,
        sequence: &[String],
    ) -> Option<SignatureMatch> {
        if sequence.is_empty() || events.len() < sequence.len() {
            return None;
        }

        // Sort events by timestamp
        let mut sorted_events: Vec<_> = events.iter().collect();
        sorted_events.sort_by_key(|e| e.timestamp);

        // Sliding window to find sequence
        let mut best_match: Option<(Vec<u64>, f64)> = None;

        for window in sorted_events.windows(sequence.len()) {
            let mut matched = true;
            let mut matched_ids = Vec::new();

            for (i, expected_type) in sequence.iter().enumerate() {
                if window[i].event_type != *expected_type {
                    matched = false;
                    break;
                }
                matched_ids.push(window[i].id);
            }

            if matched {
                // Calculate match score based on time compression
                let time_span = window.last().unwrap().timestamp - window.first().unwrap().timestamp;
                let time_score = if time_span < 300 {
                    100.0 // Very rapid sequence
                } else if time_span < 3600 {
                    80.0
                } else if time_span < 86400 {
                    60.0
                } else {
                    40.0
                };

                let score = time_score * (signature.severity / 100.0);

                if best_match.is_none() || score > best_match.as_ref().unwrap().1 {
                    best_match = Some((matched_ids, score));
                }
            }
        }

        best_match.map(|(matched_events, score)| SignatureMatch {
            signature_id: signature.id,
            signature_name: signature.name.clone(),
            score,
            matched_events,
            details: format!("Event sequence matched: {:?}", sequence),
        })
    }

    /// Match event with specific attributes.
    fn match_event_attributes(
        events: &[UserEvent],
        signature: &FraudSignature,
        event_type: &str,
        required_attrs: &HashMap<String, EventValue>,
    ) -> Option<SignatureMatch> {
        let mut matched_events = Vec::new();

        for event in events {
            if event.event_type != event_type {
                continue;
            }

            let attrs_match = required_attrs.iter().all(|(key, expected_value)| {
                event.attributes.get(key).map_or(false, |actual| {
                    Self::values_match(expected_value, actual)
                })
            });

            if attrs_match {
                matched_events.push(event.id);
            }
        }

        if matched_events.is_empty() {
            return None;
        }

        let score = signature.severity * (matched_events.len() as f64 / events.len() as f64).min(1.0);

        Some(SignatureMatch {
            signature_id: signature.id,
            signature_name: signature.name.clone(),
            score,
            matched_events,
            details: format!(
                "Events of type '{}' matched with required attributes",
                event_type
            ),
        })
    }

    /// Match events occurring within a time window.
    fn match_time_window(
        events: &[UserEvent],
        signature: &FraudSignature,
        event_types: &[String],
        window_secs: u64,
    ) -> Option<SignatureMatch> {
        if event_types.is_empty() {
            return None;
        }

        // Sort events by timestamp
        let mut sorted_events: Vec<_> = events.iter().collect();
        sorted_events.sort_by_key(|e| e.timestamp);

        let mut best_match: Option<(Vec<u64>, f64)> = None;

        // Check each potential window
        for (i, start_event) in sorted_events.iter().enumerate() {
            let window_end = start_event.timestamp + window_secs;

            // Collect events in window
            let window_events: Vec<_> = sorted_events[i..]
                .iter()
                .take_while(|e| e.timestamp <= window_end)
                .collect();

            // Check if all required event types are present
            let found_types: std::collections::HashSet<_> =
                window_events.iter().map(|e| &e.event_type).collect();

            let all_present = event_types.iter().all(|t| found_types.contains(t));

            if all_present {
                let matched_ids: Vec<_> = window_events.iter().map(|e| e.id).collect();
                let actual_span = window_events.last().unwrap().timestamp
                    - window_events.first().unwrap().timestamp;

                // Score higher for tighter windows
                let compression_ratio = 1.0 - (actual_span as f64 / window_secs as f64);
                let score = signature.severity * (0.5 + compression_ratio * 0.5);

                if best_match.is_none() || score > best_match.as_ref().unwrap().1 {
                    best_match = Some((matched_ids, score));
                }
            }
        }

        best_match.map(|(matched_events, score)| SignatureMatch {
            signature_id: signature.id,
            signature_name: signature.name.clone(),
            score,
            matched_events,
            details: format!(
                "Events {:?} found within {} second window",
                event_types, window_secs
            ),
        })
    }

    /// Match count threshold pattern.
    fn match_count_threshold(
        events: &[UserEvent],
        signature: &FraudSignature,
        event_type: &str,
        threshold: u32,
        window_secs: u64,
    ) -> Option<SignatureMatch> {
        // Filter relevant events
        let mut relevant_events: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect();

        if relevant_events.len() < threshold as usize {
            return None;
        }

        relevant_events.sort_by_key(|e| e.timestamp);

        let mut best_match: Option<(Vec<u64>, u32, f64)> = None;

        // Sliding window to find threshold breach
        for (i, start_event) in relevant_events.iter().enumerate() {
            let window_end = start_event.timestamp + window_secs;

            let window_events: Vec<_> = relevant_events[i..]
                .iter()
                .take_while(|e| e.timestamp <= window_end)
                .collect();

            let count = window_events.len() as u32;

            if count >= threshold {
                let matched_ids: Vec<_> = window_events.iter().map(|e| e.id).collect();

                // Score based on how much threshold is exceeded
                let excess_ratio = (count as f64 / threshold as f64) - 1.0;
                let score = signature.severity * (0.5 + excess_ratio.min(1.0) * 0.5);

                if best_match.is_none() || count > best_match.as_ref().unwrap().1 {
                    best_match = Some((matched_ids, count, score));
                }
            }
        }

        best_match.map(|(matched_events, count, score)| SignatureMatch {
            signature_id: signature.id,
            signature_name: signature.name.clone(),
            score,
            matched_events,
            details: format!(
                "Count threshold exceeded: {} '{}' events in {} seconds (threshold: {})",
                count, event_type, window_secs, threshold
            ),
        })
    }

    /// Match regex pattern against event data.
    fn match_regex(
        events: &[UserEvent],
        signature: &FraudSignature,
        pattern: &str,
    ) -> Option<SignatureMatch> {
        // Simple pattern matching (in production, use regex crate)
        let mut matched_events = Vec::new();

        for event in events {
            // Check event type
            if event.event_type.contains(pattern) {
                matched_events.push(event.id);
                continue;
            }

            // Check attributes
            for value in event.attributes.values() {
                if Self::value_contains_pattern(value, pattern) {
                    matched_events.push(event.id);
                    break;
                }
            }
        }

        if matched_events.is_empty() {
            return None;
        }

        let match_count = matched_events.len();
        let score = signature.severity * (match_count as f64 / events.len() as f64).min(1.0);

        Some(SignatureMatch {
            signature_id: signature.id,
            signature_name: signature.name.clone(),
            score,
            matched_events,
            details: format!("Pattern '{}' matched in {} events", pattern, match_count),
        })
    }

    /// Check if two EventValues match.
    fn values_match(expected: &EventValue, actual: &EventValue) -> bool {
        match (expected, actual) {
            (EventValue::String(e), EventValue::String(a)) => e == a,
            (EventValue::Number(e), EventValue::Number(a)) => (e - a).abs() < 0.0001,
            (EventValue::Bool(e), EventValue::Bool(a)) => e == a,
            (EventValue::List(e), EventValue::List(a)) => {
                e.len() == a.len()
                    && e.iter().zip(a.iter()).all(|(ev, av)| Self::values_match(ev, av))
            }
            _ => false,
        }
    }

    /// Check if value contains pattern string.
    fn value_contains_pattern(value: &EventValue, pattern: &str) -> bool {
        match value {
            EventValue::String(s) => s.contains(pattern),
            EventValue::List(items) => items.iter().any(|v| Self::value_contains_pattern(v, pattern)),
            _ => false,
        }
    }

    /// Get standard fraud signatures.
    pub fn standard_signatures() -> Vec<FraudSignature> {
        vec![
            FraudSignature {
                id: 1,
                name: "Rapid Login Attempts".to_string(),
                pattern: SignaturePattern::CountThreshold {
                    event_type: "login_attempt".to_string(),
                    count: 5,
                    window_secs: 60,
                },
                severity: 70.0,
                active: true,
            },
            FraudSignature {
                id: 2,
                name: "Account Takeover Sequence".to_string(),
                pattern: SignaturePattern::EventSequence(vec![
                    "password_reset".to_string(),
                    "login".to_string(),
                    "profile_change".to_string(),
                ]),
                severity: 90.0,
                active: true,
            },
            FraudSignature {
                id: 3,
                name: "Suspicious Time Window".to_string(),
                pattern: SignaturePattern::TimeWindow {
                    events: vec![
                        "login".to_string(),
                        "high_value_purchase".to_string(),
                        "logout".to_string(),
                    ],
                    window_secs: 300,
                },
                severity: 80.0,
                active: true,
            },
            FraudSignature {
                id: 4,
                name: "Gift Card Fraud Pattern".to_string(),
                pattern: SignaturePattern::Regex("gift.*card".to_string()),
                severity: 60.0,
                active: true,
            },
        ]
    }
}

impl GpuKernel for FraudSignatureDetection {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_login_attack_events() -> Vec<UserEvent> {
        let base_ts = 1700000000u64;
        (0..10)
            .map(|i| UserEvent {
                id: i as u64,
                user_id: 100,
                event_type: "login_attempt".to_string(),
                timestamp: base_ts + (i as u64 * 5), // 5 seconds apart
                attributes: HashMap::new(),
                session_id: None,
                device_id: Some("unknown".to_string()),
                ip_address: Some("10.0.0.1".to_string()),
                location: Some("XX".to_string()),
            })
            .collect()
    }

    fn create_account_takeover_events() -> Vec<UserEvent> {
        let base_ts = 1700000000u64;
        vec![
            UserEvent {
                id: 1,
                user_id: 100,
                event_type: "password_reset".to_string(),
                timestamp: base_ts,
                attributes: HashMap::new(),
                session_id: None,
                device_id: Some("new_device".to_string()),
                ip_address: None,
                location: None,
            },
            UserEvent {
                id: 2,
                user_id: 100,
                event_type: "login".to_string(),
                timestamp: base_ts + 60,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("new_device".to_string()),
                ip_address: None,
                location: None,
            },
            UserEvent {
                id: 3,
                user_id: 100,
                event_type: "profile_change".to_string(),
                timestamp: base_ts + 120,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: Some("new_device".to_string()),
                ip_address: None,
                location: None,
            },
        ]
    }

    #[test]
    fn test_fraud_signature_metadata() {
        let kernel = FraudSignatureDetection::new();
        assert_eq!(kernel.metadata().id, "behavioral/fraud-signatures");
        assert_eq!(kernel.metadata().domain, Domain::BehavioralAnalytics);
    }

    #[test]
    fn test_count_threshold_detection() {
        let events = create_login_attack_events();
        let signatures = vec![FraudSignature {
            id: 1,
            name: "Rapid Login Attempts".to_string(),
            pattern: SignaturePattern::CountThreshold {
                event_type: "login_attempt".to_string(),
                count: 5,
                window_secs: 60,
            },
            severity: 70.0,
            active: true,
        }];

        let matches = FraudSignatureDetection::compute(&events, &signatures);

        assert!(!matches.is_empty(), "Should detect rapid login pattern");
        assert_eq!(matches[0].signature_id, 1);
        assert!(matches[0].score > 50.0);
    }

    #[test]
    fn test_sequence_detection() {
        let events = create_account_takeover_events();
        let signatures = vec![FraudSignature {
            id: 2,
            name: "Account Takeover Sequence".to_string(),
            pattern: SignaturePattern::EventSequence(vec![
                "password_reset".to_string(),
                "login".to_string(),
                "profile_change".to_string(),
            ]),
            severity: 90.0,
            active: true,
        }];

        let matches = FraudSignatureDetection::compute(&events, &signatures);

        assert!(!matches.is_empty(), "Should detect account takeover sequence");
        assert_eq!(matches[0].signature_id, 2);
        assert_eq!(matches[0].matched_events.len(), 3);
    }

    #[test]
    fn test_time_window_detection() {
        let base_ts = 1700000000u64;
        let events = vec![
            UserEvent {
                id: 1,
                user_id: 100,
                event_type: "login".to_string(),
                timestamp: base_ts,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: None,
                ip_address: None,
                location: None,
            },
            UserEvent {
                id: 2,
                user_id: 100,
                event_type: "high_value_purchase".to_string(),
                timestamp: base_ts + 120,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: None,
                ip_address: None,
                location: None,
            },
            UserEvent {
                id: 3,
                user_id: 100,
                event_type: "logout".to_string(),
                timestamp: base_ts + 180,
                attributes: HashMap::new(),
                session_id: Some(1),
                device_id: None,
                ip_address: None,
                location: None,
            },
        ];

        let signatures = vec![FraudSignature {
            id: 3,
            name: "Suspicious Time Window".to_string(),
            pattern: SignaturePattern::TimeWindow {
                events: vec![
                    "login".to_string(),
                    "high_value_purchase".to_string(),
                    "logout".to_string(),
                ],
                window_secs: 300,
            },
            severity: 80.0,
            active: true,
        }];

        let matches = FraudSignatureDetection::compute(&events, &signatures);

        assert!(!matches.is_empty(), "Should detect time window pattern");
        assert_eq!(matches[0].signature_id, 3);
    }

    #[test]
    fn test_regex_detection() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "item_type".to_string(),
            EventValue::String("gift_card_100".to_string()),
        );

        let events = vec![UserEvent {
            id: 1,
            user_id: 100,
            event_type: "purchase".to_string(),
            timestamp: 1700000000,
            attributes: attrs,
            session_id: Some(1),
            device_id: None,
            ip_address: None,
            location: None,
        }];

        let signatures = vec![FraudSignature {
            id: 4,
            name: "Gift Card Fraud Pattern".to_string(),
            pattern: SignaturePattern::Regex("gift_card".to_string()),
            severity: 60.0,
            active: true,
        }];

        let matches = FraudSignatureDetection::compute(&events, &signatures);

        assert!(!matches.is_empty(), "Should detect gift card pattern");
        assert_eq!(matches[0].signature_id, 4);
    }

    #[test]
    fn test_inactive_signature_ignored() {
        let events = create_login_attack_events();
        let signatures = vec![FraudSignature {
            id: 1,
            name: "Rapid Login Attempts".to_string(),
            pattern: SignaturePattern::CountThreshold {
                event_type: "login_attempt".to_string(),
                count: 5,
                window_secs: 60,
            },
            severity: 70.0,
            active: false, // Inactive
        }];

        let matches = FraudSignatureDetection::compute(&events, &signatures);

        assert!(matches.is_empty(), "Inactive signatures should be ignored");
    }

    #[test]
    fn test_event_attributes_match() {
        let mut attrs = HashMap::new();
        attrs.insert("country".to_string(), EventValue::String("XX".to_string()));
        attrs.insert("amount".to_string(), EventValue::Number(10000.0));

        let events = vec![UserEvent {
            id: 1,
            user_id: 100,
            event_type: "transfer".to_string(),
            timestamp: 1700000000,
            attributes: attrs,
            session_id: None,
            device_id: None,
            ip_address: None,
            location: None,
        }];

        let mut required_attrs = HashMap::new();
        required_attrs.insert("country".to_string(), EventValue::String("XX".to_string()));

        let signatures = vec![FraudSignature {
            id: 5,
            name: "High Risk Country Transfer".to_string(),
            pattern: SignaturePattern::EventAttributes("transfer".to_string(), required_attrs),
            severity: 75.0,
            active: true,
        }];

        let matches = FraudSignatureDetection::compute(&events, &signatures);

        assert!(!matches.is_empty(), "Should match event attributes");
        assert_eq!(matches[0].signature_id, 5);
    }

    #[test]
    fn test_standard_signatures() {
        let signatures = FraudSignatureDetection::standard_signatures();

        assert!(!signatures.is_empty());
        assert!(signatures.iter().all(|s| s.active));
    }

    #[test]
    fn test_no_match() {
        let events = vec![UserEvent {
            id: 1,
            user_id: 100,
            event_type: "normal_activity".to_string(),
            timestamp: 1700000000,
            attributes: HashMap::new(),
            session_id: Some(1),
            device_id: None,
            ip_address: None,
            location: None,
        }];

        let signatures = FraudSignatureDetection::standard_signatures();
        let matches = FraudSignatureDetection::compute(&events, &signatures);

        assert!(matches.is_empty(), "Should not match normal activity");
    }
}
