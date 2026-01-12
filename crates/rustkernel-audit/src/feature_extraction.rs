//! Feature extraction kernel for financial audit.
//!
//! Extracts feature vectors from audit records for analysis and anomaly detection.

use crate::types::*;
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::collections::{HashMap, HashSet};

// ============================================================================
// FeatureExtraction Kernel
// ============================================================================

/// Feature extraction kernel for audit records.
///
/// Extracts numerical feature vectors from audit records for
/// machine learning analysis and anomaly detection.
#[derive(Debug, Clone)]
pub struct FeatureExtraction {
    metadata: KernelMetadata,
}

impl Default for FeatureExtraction {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtraction {
    /// Create a new feature extraction kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::batch("audit/feature-extraction", Domain::FinancialAudit)
                .with_description("Audit feature vector extraction")
                .with_throughput(50_000)
                .with_latency_us(50.0),
        }
    }

    /// Extract features from audit records.
    pub fn extract(
        records: &[AuditRecord],
        config: &FeatureConfig,
    ) -> FeatureExtractionResult {
        // Group records by entity
        let mut entity_records: HashMap<String, Vec<&AuditRecord>> = HashMap::new();
        for record in records {
            entity_records
                .entry(record.entity_id.clone())
                .or_default()
                .push(record);
        }

        // Extract features for each entity
        let mut entity_features = Vec::new();
        for (entity_id, records) in &entity_records {
            let features = Self::extract_entity_features(entity_id, records, config);
            entity_features.push(features);
        }

        // Calculate global statistics
        let global_stats = Self::calculate_global_stats(&entity_features, config);

        // Calculate anomaly scores
        let anomaly_scores = if config.detect_anomalies {
            Self::calculate_anomaly_scores(&entity_features, &global_stats)
        } else {
            HashMap::new()
        };

        FeatureExtractionResult {
            entity_features,
            global_stats,
            anomaly_scores,
        }
    }

    /// Extract features for a single entity.
    fn extract_entity_features(
        entity_id: &str,
        records: &[&AuditRecord],
        config: &FeatureConfig,
    ) -> EntityFeatureVector {
        let mut features = Vec::new();
        let mut feature_names = Vec::new();

        // Transaction volume features
        if config.include_volume_features {
            let (volume_features, volume_names) = Self::extract_volume_features(records);
            features.extend(volume_features);
            feature_names.extend(volume_names);
        }

        // Temporal features
        if config.include_temporal_features {
            let (temporal_features, temporal_names) = Self::extract_temporal_features(records);
            features.extend(temporal_features);
            feature_names.extend(temporal_names);
        }

        // Distribution features
        if config.include_distribution_features {
            let (dist_features, dist_names) = Self::extract_distribution_features(records);
            features.extend(dist_features);
            feature_names.extend(dist_names);
        }

        // Network features
        if config.include_network_features {
            let (network_features, network_names) = Self::extract_network_features(records);
            features.extend(network_features);
            feature_names.extend(network_names);
        }

        EntityFeatureVector {
            entity_id: entity_id.to_string(),
            features,
            feature_names,
            metadata: HashMap::new(),
        }
    }

    /// Extract volume-based features.
    fn extract_volume_features(records: &[&AuditRecord]) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Total transaction count
        features.push(records.len() as f64);
        names.push("total_count".to_string());

        // Total amount
        let total_amount: f64 = records.iter()
            .filter_map(|r| r.amount)
            .sum();
        features.push(total_amount);
        names.push("total_amount".to_string());

        // Average amount
        let amounts: Vec<f64> = records.iter()
            .filter_map(|r| r.amount)
            .collect();
        let avg_amount = if !amounts.is_empty() {
            total_amount / amounts.len() as f64
        } else {
            0.0
        };
        features.push(avg_amount);
        names.push("avg_amount".to_string());

        // Max amount
        let max_amount = amounts.iter().cloned().fold(0.0, f64::max);
        features.push(max_amount);
        names.push("max_amount".to_string());

        // Amount standard deviation
        let std_amount = Self::std_dev(&amounts);
        features.push(std_amount);
        names.push("std_amount".to_string());

        // Count by record type
        let mut type_counts: HashMap<AuditRecordType, usize> = HashMap::new();
        for record in records {
            *type_counts.entry(record.record_type).or_insert(0) += 1;
        }

        let record_types = [
            AuditRecordType::JournalEntry,
            AuditRecordType::Invoice,
            AuditRecordType::Payment,
            AuditRecordType::Receipt,
            AuditRecordType::Adjustment,
            AuditRecordType::Transfer,
            AuditRecordType::Expense,
            AuditRecordType::Revenue,
        ];

        for rt in record_types {
            features.push(*type_counts.get(&rt).unwrap_or(&0) as f64);
            names.push(format!("count_{:?}", rt).to_lowercase());
        }

        (features, names)
    }

    /// Extract temporal features.
    fn extract_temporal_features(records: &[&AuditRecord]) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if records.is_empty() {
            return (vec![0.0; 6], vec![
                "time_span_days".to_string(),
                "avg_interval_hours".to_string(),
                "activity_ratio".to_string(),
                "weekend_ratio".to_string(),
                "month_end_ratio".to_string(),
                "off_hours_ratio".to_string(),
            ]);
        }

        // Time span
        let timestamps: Vec<u64> = records.iter().map(|r| r.timestamp).collect();
        let min_ts = *timestamps.iter().min().unwrap_or(&0);
        let max_ts = *timestamps.iter().max().unwrap_or(&0);
        let time_span_days = (max_ts - min_ts) as f64 / 86400.0;
        features.push(time_span_days);
        names.push("time_span_days".to_string());

        // Average interval between transactions
        let mut sorted_ts = timestamps.clone();
        sorted_ts.sort();
        let avg_interval = if sorted_ts.len() > 1 {
            let intervals: Vec<f64> = sorted_ts.windows(2)
                .map(|w| (w[1] - w[0]) as f64 / 3600.0)
                .collect();
            intervals.iter().sum::<f64>() / intervals.len() as f64
        } else {
            0.0
        };
        features.push(avg_interval);
        names.push("avg_interval_hours".to_string());

        // Activity concentration
        let unique_days: HashSet<u64> = timestamps.iter()
            .map(|t| t / 86400)
            .collect();
        let activity_ratio = if time_span_days > 0.0 {
            unique_days.len() as f64 / time_span_days.max(1.0)
        } else {
            0.0
        };
        features.push(activity_ratio);
        names.push("activity_ratio".to_string());

        // Weekend activity ratio
        let weekend_count = timestamps.iter()
            .filter(|t| {
                let day_of_week = (*t / 86400) % 7;
                day_of_week == 5 || day_of_week == 6  // Simplified weekend check
            })
            .count();
        features.push(weekend_count as f64 / records.len() as f64);
        names.push("weekend_ratio".to_string());

        // Month-end activity ratio (last 5 days of month, simplified)
        let month_end_count = timestamps.iter()
            .filter(|t| {
                let day_of_month = ((*t / 86400) % 30) as u32;
                day_of_month >= 25
            })
            .count();
        features.push(month_end_count as f64 / records.len() as f64);
        names.push("month_end_ratio".to_string());

        // Off-hours activity (outside 9-17, simplified)
        let off_hours_count = timestamps.iter()
            .filter(|t| {
                let hour = ((*t / 3600) % 24) as u32;
                hour < 9 || hour >= 17
            })
            .count();
        features.push(off_hours_count as f64 / records.len() as f64);
        names.push("off_hours_ratio".to_string());

        (features, names)
    }

    /// Extract distribution features.
    fn extract_distribution_features(records: &[&AuditRecord]) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        let amounts: Vec<f64> = records.iter()
            .filter_map(|r| r.amount)
            .collect();

        if amounts.is_empty() {
            return (vec![0.0; 4], vec![
                "amount_skewness".to_string(),
                "amount_kurtosis".to_string(),
                "round_number_ratio".to_string(),
                "category_concentration".to_string(),
            ]);
        }

        // Skewness
        let skewness = Self::skewness(&amounts);
        features.push(skewness);
        names.push("amount_skewness".to_string());

        // Kurtosis
        let kurtosis = Self::kurtosis(&amounts);
        features.push(kurtosis);
        names.push("amount_kurtosis".to_string());

        // Round number ratio
        let round_count = amounts.iter()
            .filter(|a| (**a % 100.0).abs() < 0.01 || (**a % 1000.0).abs() < 0.01)
            .count();
        features.push(round_count as f64 / amounts.len() as f64);
        names.push("round_number_ratio".to_string());

        // Category concentration (HHI)
        let mut category_counts: HashMap<&str, usize> = HashMap::new();
        for record in records {
            *category_counts.entry(&record.category).or_insert(0) += 1;
        }
        let total = records.len() as f64;
        let hhi: f64 = category_counts.values()
            .map(|c| (*c as f64 / total).powi(2))
            .sum();
        features.push(hhi);
        names.push("category_concentration".to_string());

        (features, names)
    }

    /// Extract network features.
    fn extract_network_features(records: &[&AuditRecord]) -> (Vec<f64>, Vec<String>) {
        let mut features = Vec::new();
        let mut names = Vec::new();

        // Unique accounts
        let unique_accounts: HashSet<&str> = records.iter()
            .filter_map(|r| r.account.as_deref())
            .collect();
        features.push(unique_accounts.len() as f64);
        names.push("unique_accounts".to_string());

        // Unique counterparties
        let unique_counterparties: HashSet<&str> = records.iter()
            .filter_map(|r| r.counter_party.as_deref())
            .collect();
        features.push(unique_counterparties.len() as f64);
        names.push("unique_counterparties".to_string());

        // Counterparty concentration
        let mut cp_counts: HashMap<&str, usize> = HashMap::new();
        for record in records {
            if let Some(cp) = &record.counter_party {
                *cp_counts.entry(cp.as_str()).or_insert(0) += 1;
            }
        }
        let total_with_cp = cp_counts.values().sum::<usize>() as f64;
        let cp_hhi: f64 = if total_with_cp > 0.0 {
            cp_counts.values()
                .map(|c| (*c as f64 / total_with_cp).powi(2))
                .sum()
        } else {
            0.0
        };
        features.push(cp_hhi);
        names.push("counterparty_concentration".to_string());

        // Self-transactions ratio
        let self_tx_count = records.iter()
            .filter(|r| {
                r.account.as_ref() == r.counter_party.as_ref()
                    && r.account.is_some()
            })
            .count();
        features.push(self_tx_count as f64 / records.len().max(1) as f64);
        names.push("self_transaction_ratio".to_string());

        (features, names)
    }

    /// Calculate global statistics.
    fn calculate_global_stats(
        entity_features: &[EntityFeatureVector],
        _config: &FeatureConfig,
    ) -> FeatureStats {
        if entity_features.is_empty() {
            return FeatureStats {
                entity_count: 0,
                record_count: 0,
                means: Vec::new(),
                std_devs: Vec::new(),
                feature_names: Vec::new(),
            };
        }

        let feature_count = entity_features[0].features.len();
        let entity_count = entity_features.len();

        let mut means = vec![0.0; feature_count];
        let mut std_devs = vec![0.0; feature_count];

        // Calculate means
        for ef in entity_features {
            for (i, f) in ef.features.iter().enumerate() {
                means[i] += f;
            }
        }
        for m in &mut means {
            *m /= entity_count as f64;
        }

        // Calculate standard deviations
        for ef in entity_features {
            for (i, f) in ef.features.iter().enumerate() {
                std_devs[i] += (f - means[i]).powi(2);
            }
        }
        for s in &mut std_devs {
            *s = (*s / entity_count as f64).sqrt();
        }

        FeatureStats {
            entity_count,
            record_count: entity_features.iter()
                .map(|ef| ef.features.first().map(|f| *f as usize).unwrap_or(0))
                .sum(),
            means,
            std_devs,
            feature_names: entity_features[0].feature_names.clone(),
        }
    }

    /// Calculate anomaly scores using z-score method.
    fn calculate_anomaly_scores(
        entity_features: &[EntityFeatureVector],
        stats: &FeatureStats,
    ) -> HashMap<String, f64> {
        let mut scores = HashMap::new();

        for ef in entity_features {
            let mut entity_score = 0.0;
            let mut count = 0;

            for (i, f) in ef.features.iter().enumerate() {
                if i < stats.std_devs.len() && stats.std_devs[i] > 0.0 {
                    let z_score = (f - stats.means[i]).abs() / stats.std_devs[i];
                    entity_score += z_score;
                    count += 1;
                }
            }

            if count > 0 {
                scores.insert(ef.entity_id.clone(), entity_score / count as f64);
            }
        }

        scores
    }

    /// Calculate standard deviation.
    fn std_dev(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Calculate skewness.
    fn skewness(values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = Self::std_dev(values);
        if std < f64::EPSILON {
            return 0.0;
        }
        let n = values.len() as f64;
        values.iter()
            .map(|v| ((v - mean) / std).powi(3))
            .sum::<f64>() / n
    }

    /// Calculate kurtosis.
    fn kurtosis(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std = Self::std_dev(values);
        if std < f64::EPSILON {
            return 0.0;
        }
        let n = values.len() as f64;
        values.iter()
            .map(|v| ((v - mean) / std).powi(4))
            .sum::<f64>() / n - 3.0  // Excess kurtosis
    }

    /// Get feature vector for a specific entity.
    pub fn get_entity_features<'a>(
        result: &'a FeatureExtractionResult,
        entity_id: &str,
    ) -> Option<&'a EntityFeatureVector> {
        result.entity_features.iter()
            .find(|ef| ef.entity_id == entity_id)
    }

    /// Get top anomalous entities.
    pub fn top_anomalies(
        result: &FeatureExtractionResult,
        limit: usize,
    ) -> Vec<(String, f64)> {
        let mut anomalies: Vec<_> = result.anomaly_scores.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        anomalies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        anomalies.truncate(limit);
        anomalies
    }
}

impl GpuKernel for FeatureExtraction {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Feature extraction configuration.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Include volume features.
    pub include_volume_features: bool,
    /// Include temporal features.
    pub include_temporal_features: bool,
    /// Include distribution features.
    pub include_distribution_features: bool,
    /// Include network features.
    pub include_network_features: bool,
    /// Detect anomalies.
    pub detect_anomalies: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            include_volume_features: true,
            include_temporal_features: true,
            include_distribution_features: true,
            include_network_features: true,
            detect_anomalies: true,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(
        id: &str,
        entity_id: &str,
        record_type: AuditRecordType,
        amount: f64,
        timestamp: u64,
    ) -> AuditRecord {
        AuditRecord {
            id: id.to_string(),
            record_type,
            entity_id: entity_id.to_string(),
            timestamp,
            amount: Some(amount),
            currency: Some("USD".to_string()),
            account: Some(format!("ACC-{}", entity_id)),
            counter_party: Some("CP001".to_string()),
            category: "Operating".to_string(),
            attributes: HashMap::new(),
        }
    }

    fn create_test_records() -> Vec<AuditRecord> {
        vec![
            create_test_record("R001", "E001", AuditRecordType::Payment, 1000.0, 1000000),
            create_test_record("R002", "E001", AuditRecordType::Invoice, 1500.0, 1000100),
            create_test_record("R003", "E001", AuditRecordType::Payment, 500.0, 1000200),
            create_test_record("R004", "E002", AuditRecordType::Revenue, 10000.0, 1000300),
            create_test_record("R005", "E002", AuditRecordType::Expense, 3000.0, 1000400),
        ]
    }

    #[test]
    fn test_extract_features() {
        let records = create_test_records();
        let config = FeatureConfig::default();

        let result = FeatureExtraction::extract(&records, &config);

        assert_eq!(result.entity_features.len(), 2);
        assert_eq!(result.global_stats.entity_count, 2);
    }

    #[test]
    fn test_entity_features() {
        let records = create_test_records();
        let config = FeatureConfig::default();

        let result = FeatureExtraction::extract(&records, &config);

        let e001 = FeatureExtraction::get_entity_features(&result, "E001").unwrap();
        assert_eq!(e001.entity_id, "E001");
        assert!(!e001.features.is_empty());

        // E001 has 3 records
        assert_eq!(e001.features[0], 3.0); // total_count
    }

    #[test]
    fn test_volume_features() {
        let records = create_test_records();
        let config = FeatureConfig {
            include_volume_features: true,
            include_temporal_features: false,
            include_distribution_features: false,
            include_network_features: false,
            detect_anomalies: false,
        };

        let result = FeatureExtraction::extract(&records, &config);

        let e001 = FeatureExtraction::get_entity_features(&result, "E001").unwrap();
        // E001: 3 transactions, total 3000, avg 1000
        assert_eq!(e001.features[0], 3.0); // total_count
        assert_eq!(e001.features[1], 3000.0); // total_amount
        assert_eq!(e001.features[2], 1000.0); // avg_amount
    }

    #[test]
    fn test_anomaly_detection() {
        let mut records = create_test_records();
        // Add an anomalous entity
        for i in 0..10 {
            records.push(create_test_record(
                &format!("R1{}", i),
                "E003",
                AuditRecordType::Payment,
                100000.0, // Very high amounts
                1000000 + i * 100,
            ));
        }

        let config = FeatureConfig::default();
        let result = FeatureExtraction::extract(&records, &config);

        // E003 should have high anomaly score
        assert!(result.anomaly_scores.contains_key("E003"));
        let top = FeatureExtraction::top_anomalies(&result, 1);
        assert_eq!(top[0].0, "E003");
    }

    #[test]
    fn test_empty_records() {
        let records: Vec<AuditRecord> = vec![];
        let config = FeatureConfig::default();

        let result = FeatureExtraction::extract(&records, &config);

        assert!(result.entity_features.is_empty());
        assert_eq!(result.global_stats.entity_count, 0);
    }

    #[test]
    fn test_feature_names() {
        let records = create_test_records();
        let config = FeatureConfig::default();

        let result = FeatureExtraction::extract(&records, &config);

        let ef = &result.entity_features[0];
        assert_eq!(ef.features.len(), ef.feature_names.len());
        assert!(ef.feature_names.contains(&"total_count".to_string()));
        assert!(ef.feature_names.contains(&"total_amount".to_string()));
    }

    #[test]
    fn test_global_stats() {
        let records = create_test_records();
        let config = FeatureConfig::default();

        let result = FeatureExtraction::extract(&records, &config);

        assert_eq!(result.global_stats.entity_count, 2);
        assert!(!result.global_stats.means.is_empty());
        assert!(!result.global_stats.std_devs.is_empty());
    }

    #[test]
    fn test_network_features() {
        let mut records = create_test_records();
        // Add records with different counterparties
        records.push(AuditRecord {
            id: "R006".to_string(),
            record_type: AuditRecordType::Payment,
            entity_id: "E001".to_string(),
            timestamp: 1000500,
            amount: Some(500.0),
            currency: Some("USD".to_string()),
            account: Some("ACC-E001".to_string()),
            counter_party: Some("CP002".to_string()),
            category: "Operating".to_string(),
            attributes: HashMap::new(),
        });

        let config = FeatureConfig {
            include_volume_features: false,
            include_temporal_features: false,
            include_distribution_features: false,
            include_network_features: true,
            detect_anomalies: false,
        };

        let result = FeatureExtraction::extract(&records, &config);

        let e001 = FeatureExtraction::get_entity_features(&result, "E001").unwrap();
        // E001 now has 2 unique counterparties
        assert!(e001.features[1] >= 2.0); // unique_counterparties
    }

    #[test]
    fn test_selective_features() {
        let records = create_test_records();

        // Volume only
        let config_vol = FeatureConfig {
            include_volume_features: true,
            include_temporal_features: false,
            include_distribution_features: false,
            include_network_features: false,
            detect_anomalies: false,
        };
        let result_vol = FeatureExtraction::extract(&records, &config_vol);

        // All features
        let config_all = FeatureConfig::default();
        let result_all = FeatureExtraction::extract(&records, &config_all);

        // All features should have more features
        assert!(result_all.entity_features[0].features.len()
            > result_vol.entity_features[0].features.len());
    }
}
