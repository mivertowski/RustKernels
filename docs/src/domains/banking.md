# Banking

**Crate**: `rustkernel-banking`
**Kernels**: 1
**Feature**: `banking`

Specialized banking operations kernel for fraud detection.

## Kernel Overview

| Kernel | ID | Modes | Description |
|--------|-----|-------|-------------|
| FraudPatternMatch | `banking/fraud-pattern-match` | Batch, Ring | Real-time fraud detection |

---

## Kernel Details

### FraudPatternMatch

GPU-accelerated fraud pattern detection using behavioral analysis and rule matching.

**ID**: `banking/fraud-pattern-match`
**Modes**: Batch, Ring
**Throughput**: ~500,000 transactions/sec

#### Input

```rust
pub struct FraudPatternInput {
    /// Transactions to analyze
    pub transactions: Vec<Transaction>,
    /// Pattern rules to apply
    pub rules: Vec<FraudRule>,
    /// Historical behavior profiles
    pub profiles: HashMap<String, BehaviorProfile>,
    /// Detection threshold
    pub threshold: f64,
}

pub struct Transaction {
    pub id: String,
    pub account_id: String,
    pub amount: f64,
    pub currency: String,
    pub merchant_category: String,
    pub location: Option<GeoLocation>,
    pub timestamp: u64,
    pub channel: TransactionChannel,
}

pub struct BehaviorProfile {
    pub avg_transaction_amount: f64,
    pub typical_merchants: Vec<String>,
    pub typical_locations: Vec<GeoLocation>,
    pub typical_hours: Vec<u8>,
}

pub enum TransactionChannel {
    CardPresent,
    CardNotPresent,
    ATM,
    Wire,
    ACH,
}
```

#### Output

```rust
pub struct FraudPatternOutput {
    /// Fraud scores per transaction
    pub scores: Vec<TransactionScore>,
    /// Triggered rules
    pub triggered_rules: Vec<TriggeredRule>,
    /// Recommended actions
    pub actions: Vec<RecommendedAction>,
}

pub struct TransactionScore {
    pub transaction_id: String,
    pub fraud_score: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub recommendation: Recommendation,
}

pub enum Recommendation {
    Approve,
    Review,
    Decline,
    Challenge,  // 3D Secure, OTP, etc.
}
```

#### Example

```rust
use rustkernel::banking::fraud::{FraudPatternMatch, FraudPatternInput};

let kernel = FraudPatternMatch::new();

let result = kernel.execute(FraudPatternInput {
    transactions: vec![
        Transaction {
            id: "TX001".into(),
            account_id: "ACC123".into(),
            amount: 5000.0,
            currency: "USD".into(),
            merchant_category: "ELECTRONICS".into(),
            location: Some(GeoLocation { lat: 40.7128, lon: -74.0060 }),
            timestamp: 1699000000,
            channel: TransactionChannel::CardNotPresent,
        },
    ],
    rules: default_rules(),
    profiles: customer_profiles,
    threshold: 0.7,
}).await?;

for score in result.scores {
    if score.fraud_score > 0.7 {
        println!("High risk: {} - score {:.2}",
            score.transaction_id,
            score.fraud_score
        );
        for factor in score.risk_factors {
            println!("  - {:?}", factor);
        }
    }
}
```

---

## Detection Methods

### Behavioral Analysis

Compares transactions against established customer behavior:

- **Amount deviation**: Transaction size vs historical average
- **Location deviation**: Geographic distance from typical locations
- **Time deviation**: Transaction time vs typical activity hours
- **Merchant deviation**: New merchant category or type

### Rule-Based Detection

Configurable rules for known fraud patterns:

```rust
pub struct FraudRule {
    pub id: String,
    pub name: String,
    pub conditions: Vec<RuleCondition>,
    pub score_impact: f64,
    pub enabled: bool,
}

pub enum RuleCondition {
    AmountGreaterThan(f64),
    VelocityExceeds { count: u32, window_seconds: u64 },
    LocationMismatch { max_distance_km: f64 },
    NewMerchant,
    HighRiskCountry(Vec<String>),
    CardNotPresentHighValue,
}
```

### Velocity Checks

Detects rapid transaction patterns:

- Multiple transactions in short time
- Multiple cards on same device
- Multiple devices for same card

---

## Ring Mode for Real-Time Scoring

Ring mode enables sub-millisecond fraud scoring:

```rust
use rustkernel::banking::fraud::FraudPatternRing;

let ring = FraudPatternRing::new();

// Pre-load customer profiles
ring.load_profiles(profiles).await?;

// Score transactions in real-time
async fn score_transaction(tx: Transaction) -> FraudDecision {
    let score = ring.score(tx).await?;

    match score.fraud_score {
        s if s < 0.3 => FraudDecision::Approve,
        s if s < 0.7 => FraudDecision::Review,
        _ => FraudDecision::Decline,
    }
}
```

---

## Integration Patterns

### Authorization Flow

```rust
// In payment authorization path
async fn authorize(tx: Transaction) -> AuthResult {
    // 1. Real-time fraud scoring (< 100ms)
    let fraud_result = fraud_ring.score(tx.clone()).await?;

    if fraud_result.recommendation == Recommendation::Decline {
        return AuthResult::Declined("Fraud risk");
    }

    if fraud_result.recommendation == Recommendation::Challenge {
        return AuthResult::Challenge3DS;
    }

    // 2. Continue with authorization
    process_authorization(tx).await
}
```

### Batch Analysis

```rust
// Daily fraud pattern review
async fn daily_fraud_review() {
    let transactions = load_day_transactions().await?;

    let result = fraud_kernel.execute(FraudPatternInput {
        transactions,
        rules: all_rules(),
        profiles: all_profiles(),
        threshold: 0.5,  // Lower threshold for review
    }).await?;

    // Generate suspicious activity report
    generate_sar_report(result.scores.filter(|s| s.fraud_score > 0.5));
}
```

## Performance Considerations

1. **Profile caching**: Keep frequently accessed profiles in GPU memory
2. **Rule optimization**: Order rules by selectivity (most filtering first)
3. **Batch when possible**: Process multiple transactions per GPU call
4. **Async patterns**: Don't block authorization on slow operations
