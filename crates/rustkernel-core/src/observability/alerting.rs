//! Alert Rules and Routing
//!
//! Defines alert rules for kernel health and performance monitoring.
//!
//! # Features
//!
//! - Alert rule definition with conditions
//! - Severity levels and routing
//! - SLO violation alerts
//! - Integration with external alerting systems
//!
//! # Example
//!
//! ```rust,ignore
//! use rustkernel_core::observability::alerting::{AlertRule, AlertSeverity, AlertConfig};
//!
//! let config = AlertConfig::default()
//!     .add_rule(
//!         AlertRule::new("high_latency")
//!             .condition("avg_latency_ms > 100")
//!             .severity(AlertSeverity::Warning)
//!             .for_duration(Duration::from_secs(60))
//!     );
//! ```

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning - may need attention
    Warning,
    /// Critical - needs immediate attention
    Critical,
    /// Page - wake someone up
    Page,
}

impl Default for AlertSeverity {
    fn default() -> Self {
        Self::Warning
    }
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
            Self::Page => write!(f, "page"),
        }
    }
}

/// Alert state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertState {
    /// Alert is not firing
    Ok,
    /// Alert condition is pending (within for_duration)
    Pending,
    /// Alert is firing
    Firing,
    /// Alert has been acknowledged
    Acknowledged,
    /// Alert has been resolved
    Resolved,
}

impl Default for AlertState {
    fn default() -> Self {
        Self::Ok
    }
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Alert routing configuration
    pub routing: AlertRouting,
    /// Evaluation interval
    pub evaluation_interval: Duration,
    /// Resolve timeout (auto-resolve after this duration of OK)
    pub resolve_timeout: Duration,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: Vec::new(),
            routing: AlertRouting::default(),
            evaluation_interval: Duration::from_secs(15),
            resolve_timeout: Duration::from_secs(300),
        }
    }
}

impl AlertConfig {
    /// Add an alert rule
    pub fn add_rule(mut self, rule: AlertRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Set routing configuration
    pub fn with_routing(mut self, routing: AlertRouting) -> Self {
        self.routing = routing;
        self
    }

    /// Set evaluation interval
    pub fn with_evaluation_interval(mut self, interval: Duration) -> Self {
        self.evaluation_interval = interval;
        self
    }

    /// Add default kernel health rules
    pub fn with_default_rules(mut self) -> Self {
        self.rules.push(AlertRule::kernel_unhealthy());
        self.rules.push(AlertRule::high_latency());
        self.rules.push(AlertRule::high_error_rate());
        self.rules.push(AlertRule::queue_depth());
        self.rules.push(AlertRule::gpu_memory());
        self
    }
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Alert condition expression
    pub condition: String,
    /// Severity level
    pub severity: AlertSeverity,
    /// Duration condition must be true before firing
    pub for_duration: Duration,
    /// Labels for routing
    pub labels: std::collections::HashMap<String, String>,
    /// Annotations for alert message
    pub annotations: std::collections::HashMap<String, String>,
    /// Kernels this rule applies to (empty = all)
    pub kernel_filter: Vec<String>,
    /// Domains this rule applies to (empty = all)
    pub domain_filter: Vec<String>,
}

impl AlertRule {
    /// Create a new alert rule
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            condition: String::new(),
            severity: AlertSeverity::Warning,
            for_duration: Duration::from_secs(0),
            labels: std::collections::HashMap::new(),
            annotations: std::collections::HashMap::new(),
            kernel_filter: Vec::new(),
            domain_filter: Vec::new(),
        }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set condition
    pub fn condition(mut self, cond: impl Into<String>) -> Self {
        self.condition = cond.into();
        self
    }

    /// Set severity
    pub fn severity(mut self, severity: AlertSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set for_duration
    pub fn for_duration(mut self, duration: Duration) -> Self {
        self.for_duration = duration;
        self
    }

    /// Add a label
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Add an annotation
    pub fn annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }

    /// Filter to specific kernels
    pub fn for_kernels(mut self, kernels: Vec<String>) -> Self {
        self.kernel_filter = kernels;
        self
    }

    /// Filter to specific domains
    pub fn for_domains(mut self, domains: Vec<String>) -> Self {
        self.domain_filter = domains;
        self
    }

    // Predefined rules

    /// Kernel unhealthy rule
    pub fn kernel_unhealthy() -> Self {
        Self::new("KernelUnhealthy")
            .description("Kernel is reporting unhealthy status")
            .condition("health_status != healthy")
            .severity(AlertSeverity::Critical)
            .for_duration(Duration::from_secs(30))
            .annotation("summary", "Kernel {{ $labels.kernel_id }} is unhealthy")
    }

    /// High latency rule
    pub fn high_latency() -> Self {
        Self::new("KernelHighLatency")
            .description("Kernel message latency is above threshold")
            .condition("avg_latency_ms > 100")
            .severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(60))
            .annotation("summary", "Kernel {{ $labels.kernel_id }} has high latency ({{ $value }}ms)")
    }

    /// High error rate rule
    pub fn high_error_rate() -> Self {
        Self::new("KernelHighErrorRate")
            .description("Kernel error rate is above threshold")
            .condition("error_rate > 0.01")
            .severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(300))
            .annotation("summary", "Kernel {{ $labels.kernel_id }} has high error rate ({{ $value }})")
    }

    /// Queue depth rule
    pub fn queue_depth() -> Self {
        Self::new("KernelQueueDepth")
            .description("Kernel message queue is getting full")
            .condition("queue_depth > 1000")
            .severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(60))
            .annotation("summary", "Kernel {{ $labels.kernel_id }} queue depth is high ({{ $value }})")
    }

    /// GPU memory rule
    pub fn gpu_memory() -> Self {
        Self::new("GPUMemoryHigh")
            .description("GPU memory usage is above 90%")
            .condition("gpu_memory_percent > 90")
            .severity(AlertSeverity::Critical)
            .for_duration(Duration::from_secs(60))
            .annotation("summary", "GPU memory usage is critically high ({{ $value }}%)")
    }

    /// SLO violation rule
    pub fn slo_violation(slo_name: impl Into<String>) -> Self {
        let name = slo_name.into();
        Self::new(format!("SLOViolation_{}", name))
            .description(format!("SLO '{}' is being violated", name))
            .condition(format!("slo_{}_compliance < target", name))
            .severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(300))
            .label("slo", name.clone())
            .annotation("summary", format!("SLO '{}' compliance is below target", name))
    }
}

/// Alert routing configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlertRouting {
    /// Default receiver
    pub default_receiver: Option<String>,
    /// Routes based on labels
    pub routes: Vec<AlertRoute>,
    /// Receiver configurations
    pub receivers: Vec<AlertReceiver>,
}

impl AlertRouting {
    /// Add a route
    pub fn add_route(mut self, route: AlertRoute) -> Self {
        self.routes.push(route);
        self
    }

    /// Add a receiver
    pub fn add_receiver(mut self, receiver: AlertReceiver) -> Self {
        self.receivers.push(receiver);
        self
    }

    /// Set default receiver
    pub fn with_default(mut self, receiver: impl Into<String>) -> Self {
        self.default_receiver = Some(receiver.into());
        self
    }
}

/// Alert route
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRoute {
    /// Label matchers
    pub matchers: std::collections::HashMap<String, String>,
    /// Receiver name
    pub receiver: String,
    /// Continue matching after this route
    pub continue_matching: bool,
    /// Group by labels
    pub group_by: Vec<String>,
    /// Group wait duration
    pub group_wait: Duration,
    /// Group interval
    pub group_interval: Duration,
}

impl AlertRoute {
    /// Create a new route
    pub fn new(receiver: impl Into<String>) -> Self {
        Self {
            matchers: std::collections::HashMap::new(),
            receiver: receiver.into(),
            continue_matching: false,
            group_by: Vec::new(),
            group_wait: Duration::from_secs(30),
            group_interval: Duration::from_secs(300),
        }
    }

    /// Add a matcher
    pub fn match_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.matchers.insert(key.into(), value.into());
        self
    }

    /// Set group by
    pub fn group_by(mut self, labels: Vec<String>) -> Self {
        self.group_by = labels;
        self
    }
}

/// Alert receiver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertReceiver {
    /// Receiver name
    pub name: String,
    /// Receiver type
    pub receiver_type: ReceiverType,
}

impl AlertReceiver {
    /// Create a new receiver
    pub fn new(name: impl Into<String>, receiver_type: ReceiverType) -> Self {
        Self {
            name: name.into(),
            receiver_type,
        }
    }

    /// Slack receiver
    pub fn slack(name: impl Into<String>, webhook_url: impl Into<String>) -> Self {
        Self::new(name, ReceiverType::Slack {
            webhook_url: webhook_url.into(),
            channel: None,
        })
    }

    /// PagerDuty receiver
    pub fn pagerduty(name: impl Into<String>, service_key: impl Into<String>) -> Self {
        Self::new(name, ReceiverType::PagerDuty {
            service_key: service_key.into(),
        })
    }

    /// Webhook receiver
    pub fn webhook(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(name, ReceiverType::Webhook { url: url.into() })
    }
}

/// Receiver type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReceiverType {
    /// Slack webhook
    Slack {
        webhook_url: String,
        channel: Option<String>,
    },
    /// PagerDuty
    PagerDuty {
        service_key: String,
    },
    /// Generic webhook
    Webhook {
        url: String,
    },
    /// Email
    Email {
        to: Vec<String>,
        from: String,
        smtp_server: String,
    },
    /// Log only (for testing)
    Log,
}

/// An active alert instance
#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    /// Alert rule name
    pub rule_name: String,
    /// Current state
    pub state: AlertState,
    /// Severity
    pub severity: AlertSeverity,
    /// Labels
    pub labels: std::collections::HashMap<String, String>,
    /// Annotations
    pub annotations: std::collections::HashMap<String, String>,
    /// When the alert started firing
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// When the alert was last updated
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Current value that triggered the alert
    pub value: Option<f64>,
}

impl Alert {
    /// Create a new alert
    pub fn new(rule: &AlertRule) -> Self {
        Self {
            rule_name: rule.name.clone(),
            state: AlertState::Pending,
            severity: rule.severity,
            labels: rule.labels.clone(),
            annotations: rule.annotations.clone(),
            started_at: None,
            updated_at: chrono::Utc::now(),
            value: None,
        }
    }

    /// Transition to firing state
    pub fn fire(&mut self) {
        if self.state != AlertState::Firing {
            self.state = AlertState::Firing;
            self.started_at = Some(chrono::Utc::now());
        }
        self.updated_at = chrono::Utc::now();
    }

    /// Transition to resolved state
    pub fn resolve(&mut self) {
        self.state = AlertState::Resolved;
        self.updated_at = chrono::Utc::now();
    }

    /// Acknowledge the alert
    pub fn acknowledge(&mut self) {
        self.state = AlertState::Acknowledged;
        self.updated_at = chrono::Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_rule() {
        let rule = AlertRule::new("test_rule")
            .description("Test rule")
            .condition("error_rate > 0.01")
            .severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(60));

        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_predefined_rules() {
        let unhealthy = AlertRule::kernel_unhealthy();
        assert_eq!(unhealthy.severity, AlertSeverity::Critical);

        let high_latency = AlertRule::high_latency();
        assert_eq!(high_latency.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_config() {
        let config = AlertConfig::default().with_default_rules();
        assert!(!config.rules.is_empty());
    }

    #[test]
    fn test_alert_state() {
        let rule = AlertRule::kernel_unhealthy();
        let mut alert = Alert::new(&rule);

        assert_eq!(alert.state, AlertState::Pending);

        alert.fire();
        assert_eq!(alert.state, AlertState::Firing);
        assert!(alert.started_at.is_some());

        alert.acknowledge();
        assert_eq!(alert.state, AlertState::Acknowledged);

        alert.resolve();
        assert_eq!(alert.state, AlertState::Resolved);
    }

    #[test]
    fn test_receivers() {
        let slack = AlertReceiver::slack("slack-ops", "https://hooks.slack.com/xxx");
        assert_eq!(slack.name, "slack-ops");

        let pagerduty = AlertReceiver::pagerduty("pagerduty-ops", "service-key");
        assert_eq!(pagerduty.name, "pagerduty-ops");
    }
}
