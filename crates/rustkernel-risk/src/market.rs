//! Market risk kernels.
//!
//! This module provides market risk analytics:
//! - Monte Carlo Value at Risk (VaR)
//! - Portfolio risk aggregation
//! - Expected Shortfall (CVaR)

use crate::messages::{
    MonteCarloVaRInput, MonteCarloVaROutput, PortfolioRiskAggregationInput,
    PortfolioRiskAggregationOutput,
};
use crate::ring_messages::{
    K2KMarketUpdate, K2KMarketUpdateAck, K2KVaRAggregation, K2KVaRAggregationResponse,
    QueryVaRResponse, QueryVaRRing, RecalculateVaRResponse, RecalculateVaRRing,
    UpdatePositionResponse, UpdatePositionRing, from_currency_fp, from_fixed_point, to_currency_fp,
};
use crate::types::{Portfolio, PortfolioRiskResult, VaRParams, VaRResult};
use async_trait::async_trait;
use ringkernel_core::RingContext;
use rustkernel_core::error::Result;
use rustkernel_core::traits::{BatchKernel, RingKernelHandler};
use rustkernel_core::{domain::Domain, kernel::KernelMetadata, traits::GpuKernel};
use std::time::Instant;

// ============================================================================
// Monte Carlo VaR Kernel
// ============================================================================

/// Monte Carlo VaR state for Ring mode operations.
#[derive(Debug, Clone, Default)]
pub struct MonteCarloVaRState {
    /// Current portfolio.
    pub portfolio: Option<Portfolio>,
    /// Cached VaR result.
    pub var: f64,
    /// Cached Expected Shortfall.
    pub es: f64,
    /// Confidence level used for cached result.
    pub confidence_level: f64,
    /// Holding period used for cached result.
    pub holding_period: u32,
    /// Whether cached result is stale.
    pub is_stale: bool,
    /// Number of simulations used.
    pub n_simulations: u32,
}

/// Monte Carlo Value at Risk kernel.
///
/// Simulates portfolio P&L using correlated random variates.
#[derive(Debug)]
pub struct MonteCarloVaR {
    metadata: KernelMetadata,
    /// Internal state for Ring mode operations.
    state: std::sync::RwLock<MonteCarloVaRState>,
}

impl Clone for MonteCarloVaR {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            state: std::sync::RwLock::new(self.state.read().unwrap().clone()),
        }
    }
}

impl Default for MonteCarloVaR {
    fn default() -> Self {
        Self::new()
    }
}

impl MonteCarloVaR {
    /// Create a new Monte Carlo VaR kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("risk/monte-carlo-var", Domain::RiskAnalytics)
                .with_description("Monte Carlo Value at Risk simulation")
                .with_throughput(100_000)
                .with_latency_us(1000.0),
            state: std::sync::RwLock::new(MonteCarloVaRState::default()),
        }
    }

    /// Initialize the kernel with a portfolio for Ring mode operations.
    pub fn initialize(&self, portfolio: Portfolio) {
        let mut state = self.state.write().unwrap();
        state.portfolio = Some(portfolio);
        state.is_stale = true;
    }

    /// Update a position value in the portfolio.
    pub fn update_position(&self, asset_id: u64, new_value: f64) -> bool {
        let mut state = self.state.write().unwrap();
        if let Some(ref mut portfolio) = state.portfolio {
            // Find and update the asset
            for (i, &id) in portfolio.asset_ids.iter().enumerate() {
                if id == asset_id {
                    portfolio.values[i] = new_value;
                    state.is_stale = true;
                    return true;
                }
            }
        }
        false
    }

    /// Get cached VaR result.
    pub fn cached_var(&self) -> (f64, f64, bool) {
        let state = self.state.read().unwrap();
        (state.var, state.es, !state.is_stale)
    }

    /// Recalculate VaR using given parameters.
    pub fn recalculate(
        &self,
        confidence_level: f64,
        holding_period: u32,
        n_simulations: u32,
    ) -> (f64, f64) {
        let mut state = self.state.write().unwrap();
        let Some(ref portfolio) = state.portfolio else {
            return (0.0, 0.0);
        };

        let params = VaRParams::new(confidence_level, holding_period, n_simulations);
        let result = Self::compute(portfolio, params);

        state.var = result.var;
        state.es = result.expected_shortfall;
        state.confidence_level = confidence_level;
        state.holding_period = holding_period;
        state.n_simulations = n_simulations;
        state.is_stale = false;

        (result.var, result.expected_shortfall)
    }

    /// Calculate VaR using Monte Carlo simulation.
    ///
    /// # Arguments
    /// * `portfolio` - Portfolio to analyze
    /// * `params` - VaR calculation parameters
    pub fn compute(portfolio: &Portfolio, params: VaRParams) -> VaRResult {
        if portfolio.n_assets() == 0 {
            return VaRResult {
                var: 0.0,
                expected_shortfall: 0.0,
                confidence_level: params.confidence_level,
                holding_period: params.holding_period,
                component_var: Vec::new(),
                marginal_var: Vec::new(),
                percentiles: Vec::new(),
            };
        }

        let n_sims = params.n_simulations as usize;
        let holding_factor = (params.holding_period as f64).sqrt();

        // Generate correlated random returns using Cholesky decomposition
        let cholesky = Self::cholesky_decomposition(portfolio);

        // Simulate P&L scenarios
        let mut pnl_scenarios = Vec::with_capacity(n_sims);
        let mut rng = SimpleRng::new(42); // Deterministic for reproducibility

        for _ in 0..n_sims {
            // Generate independent standard normal variates
            let z: Vec<f64> = (0..portfolio.n_assets()).map(|_| rng.normal()).collect();

            // Correlate using Cholesky
            let correlated = Self::apply_cholesky(&cholesky, &z, portfolio.n_assets());

            // Calculate portfolio P&L
            let mut scenario_pnl = 0.0;
            for (i, (&z_corr, (&vol, &value))) in correlated
                .iter()
                .zip(portfolio.volatilities.iter().zip(portfolio.values.iter()))
                .enumerate()
            {
                let ret = portfolio.expected_returns[i] * params.holding_period as f64 / 252.0
                    + vol * holding_factor / (252.0_f64).sqrt() * z_corr;
                scenario_pnl += value * ret;
            }

            pnl_scenarios.push(scenario_pnl);
        }

        // Sort scenarios for percentile calculation
        pnl_scenarios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate VaR (loss at confidence level)
        let var_idx = ((1.0 - params.confidence_level) * n_sims as f64) as usize;
        let var = -pnl_scenarios[var_idx.min(n_sims - 1)]; // Positive VaR = loss

        // Calculate Expected Shortfall (average of tail losses)
        let tail_start = var_idx.min(n_sims - 1);
        let expected_shortfall = if tail_start > 0 {
            -pnl_scenarios[..tail_start].iter().sum::<f64>() / tail_start as f64
        } else {
            var
        };

        // Calculate component VaR (marginal contribution)
        let component_var = Self::calculate_component_var(portfolio, var, &cholesky);

        // Calculate marginal VaR
        let marginal_var = Self::calculate_marginal_var(portfolio, params, &cholesky);

        // Calculate percentiles
        let percentiles: Vec<(f64, f64)> = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            .iter()
            .map(|&p| {
                let idx = ((1.0 - p) * n_sims as f64) as usize;
                (p, pnl_scenarios[idx.min(n_sims - 1)])
            })
            .collect();

        VaRResult {
            var,
            expected_shortfall,
            confidence_level: params.confidence_level,
            holding_period: params.holding_period,
            component_var,
            marginal_var,
            percentiles,
        }
    }

    /// Perform Cholesky decomposition of correlation matrix.
    fn cholesky_decomposition(portfolio: &Portfolio) -> Vec<f64> {
        let n = portfolio.n_assets();
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }

                if i == j {
                    let diag = portfolio.correlation(i, i) - sum;
                    l[i * n + j] = if diag > 0.0 { diag.sqrt() } else { 0.0 };
                } else {
                    let l_jj = l[j * n + j];
                    l[i * n + j] = if l_jj.abs() > 1e-10 {
                        (portfolio.correlation(i, j) - sum) / l_jj
                    } else {
                        0.0
                    };
                }
            }
        }

        l
    }

    /// Apply Cholesky factor to independent normals.
    fn apply_cholesky(l: &[f64], z: &[f64], n: usize) -> Vec<f64> {
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                result[i] += l[i * n + j] * z[j];
            }
        }
        result
    }

    /// Calculate component VaR.
    #[allow(clippy::needless_range_loop)]
    fn calculate_component_var(
        portfolio: &Portfolio,
        total_var: f64,
        _cholesky: &[f64],
    ) -> Vec<f64> {
        // Component VaR = weight_i * sigma_i * rho_i,p * VaR / sigma_p
        let weights = portfolio.weights();
        let n = portfolio.n_assets();

        // Calculate portfolio volatility
        let port_var = Self::portfolio_variance(portfolio);
        let port_vol = port_var.sqrt();

        if port_vol < 1e-10 {
            return vec![0.0; n];
        }

        // Calculate covariance of each asset with portfolio
        let mut component_vars = Vec::with_capacity(n);

        for i in 0..n {
            let mut cov_i_p = 0.0;
            for j in 0..n {
                cov_i_p += weights[j]
                    * portfolio.volatilities[i]
                    * portfolio.volatilities[j]
                    * portfolio.correlation(i, j);
            }

            let beta_i = cov_i_p / port_var;
            let component_var_i = weights[i] * beta_i * total_var;
            component_vars.push(component_var_i);
        }

        component_vars
    }

    /// Calculate marginal VaR.
    #[allow(clippy::needless_range_loop)]
    fn calculate_marginal_var(
        portfolio: &Portfolio,
        params: VaRParams,
        _cholesky: &[f64],
    ) -> Vec<f64> {
        let n = portfolio.n_assets();
        let weights = portfolio.weights();
        let holding_factor = (params.holding_period as f64).sqrt();

        // Z-score for confidence level
        let z = Self::norm_inv(params.confidence_level);

        // Portfolio volatility
        let port_vol = Self::portfolio_variance(portfolio).sqrt();

        if port_vol < 1e-10 {
            return vec![0.0; n];
        }

        // Marginal VaR = dVaR/dw_i = z * sigma_p * d_sigma_p/dw_i
        let mut marginal_vars = Vec::with_capacity(n);

        for i in 0..n {
            let mut cov_i_p = 0.0;
            for j in 0..n {
                cov_i_p += weights[j]
                    * portfolio.volatilities[i]
                    * portfolio.volatilities[j]
                    * portfolio.correlation(i, j);
            }

            let d_sigma_dw = cov_i_p / port_vol;
            let marginal_var_i = z * d_sigma_dw * holding_factor / (252.0_f64).sqrt();
            marginal_vars.push(marginal_var_i * portfolio.total_value());
        }

        marginal_vars
    }

    /// Calculate portfolio variance.
    fn portfolio_variance(portfolio: &Portfolio) -> f64 {
        let weights = portfolio.weights();
        let n = portfolio.n_assets();
        let mut var = 0.0;

        for i in 0..n {
            for j in 0..n {
                var += weights[i]
                    * weights[j]
                    * portfolio.volatilities[i]
                    * portfolio.volatilities[j]
                    * portfolio.correlation(i, j);
            }
        }

        var
    }

    /// Standard normal inverse CDF.
    fn norm_inv(p: f64) -> f64 {
        // Rational approximation
        let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
        let t = if p_clamped < 0.5 {
            (-2.0 * p_clamped.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p_clamped).ln()).sqrt()
        };

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p_clamped < 0.5 { -result } else { result }
    }
}

impl GpuKernel for MonteCarloVaR {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

// ============================================================================
// Monte Carlo VaR RingKernelHandler Implementations
// ============================================================================

/// RingKernelHandler for position updates.
///
/// Enables streaming position updates for real-time VaR.
#[async_trait]
impl RingKernelHandler<UpdatePositionRing, UpdatePositionResponse> for MonteCarloVaR {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: UpdatePositionRing,
    ) -> Result<UpdatePositionResponse> {
        // Update position in internal state
        let new_value = from_currency_fp(msg.value_fp);
        let updated = self.update_position(msg.asset_id, new_value);

        // Position changed, VaR needs recalculation
        Ok(UpdatePositionResponse {
            request_id: msg.id.0,
            asset_id: msg.asset_id,
            var_stale: updated,
        })
    }
}

/// RingKernelHandler for VaR queries.
///
/// Returns current cached VaR or triggers recalculation.
#[async_trait]
impl RingKernelHandler<QueryVaRRing, QueryVaRResponse> for MonteCarloVaR {
    async fn handle(&self, _ctx: &mut RingContext, msg: QueryVaRRing) -> Result<QueryVaRResponse> {
        // Query cached VaR from internal state
        let (var, es, is_fresh) = self.cached_var();

        Ok(QueryVaRResponse {
            request_id: msg.id.0,
            var_fp: to_currency_fp(var),
            es_fp: to_currency_fp(es),
            confidence_fp: msg.confidence_fp,
            holding_period: msg.holding_period,
            is_fresh,
        })
    }
}

/// RingKernelHandler for VaR recalculation.
///
/// Triggers full Monte Carlo simulation with given parameters.
#[async_trait]
impl RingKernelHandler<RecalculateVaRRing, RecalculateVaRResponse> for MonteCarloVaR {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: RecalculateVaRRing,
    ) -> Result<RecalculateVaRResponse> {
        let start = Instant::now();
        let confidence = from_fixed_point(msg.confidence_fp);

        // Run full Monte Carlo simulation on internal state
        let (var, es) = self.recalculate(confidence, msg.holding_period, msg.n_simulations);
        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(RecalculateVaRResponse {
            request_id: msg.id.0,
            var_fp: to_currency_fp(var),
            es_fp: to_currency_fp(es),
            compute_time_us,
            n_simulations: msg.n_simulations,
        })
    }
}

/// RingKernelHandler for K2K market updates.
///
/// Processes streaming market data for real-time VaR adjustments.
#[async_trait]
impl RingKernelHandler<K2KMarketUpdate, K2KMarketUpdateAck> for MonteCarloVaR {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: K2KMarketUpdate,
    ) -> Result<K2KMarketUpdateAck> {
        // Get current portfolio value for impact estimation
        let portfolio_value = {
            let state = self.state.read().unwrap();
            state
                .portfolio
                .as_ref()
                .map(|p| p.total_value())
                .unwrap_or(0.0)
        };

        // Estimate VaR impact from volatility change
        // VaR impact ≈ portfolio_value * z * Δvolatility * sqrt(holding_period)
        let vol_delta = from_fixed_point(msg.vol_delta_fp);
        let z_95 = 1.645; // 95% confidence z-score
        let var_impact = portfolio_value * z_95 * vol_delta.abs();

        // Mark state as stale since market conditions changed
        {
            let mut state = self.state.write().unwrap();
            state.is_stale = true;
        }

        Ok(K2KMarketUpdateAck {
            request_id: msg.id.0,
            worker_id: 0, // Single instance worker
            var_impact_fp: to_currency_fp(var_impact),
        })
    }
}

/// RingKernelHandler for K2K partial VaR aggregation.
///
/// Used in distributed VaR calculation to aggregate partial results.
#[async_trait]
impl RingKernelHandler<K2KVaRAggregation, K2KVaRAggregationResponse> for MonteCarloVaR {
    async fn handle(
        &self,
        _ctx: &mut RingContext,
        msg: K2KVaRAggregation,
    ) -> Result<K2KVaRAggregationResponse> {
        let complete = msg.workers_reported >= msg.expected_workers;
        let aggregated_var = from_currency_fp(msg.aggregated_var_fp);

        // Calculate diversification benefit based on portfolio correlation structure
        // With n workers, diversification benefit scales with 1 - sqrt(1/n) for uncorrelated
        // For partially correlated, use average correlation from portfolio if available
        let diversification_factor = {
            let state = self.state.read().unwrap();
            if let Some(ref portfolio) = state.portfolio {
                let n = portfolio.n_assets();
                if n > 1 {
                    // Calculate average off-diagonal correlation
                    let mut avg_corr = 0.0;
                    let mut count = 0;
                    for i in 0..n {
                        for j in (i + 1)..n {
                            avg_corr += portfolio.correlation(i, j);
                            count += 1;
                        }
                    }
                    if count > 0 {
                        avg_corr /= count as f64;
                    }
                    // Diversification benefit = 1 - sqrt(average correlation adjustment)
                    // Higher correlation = less benefit
                    (1.0 - avg_corr.max(0.0).sqrt()) * 0.3 // Cap at 30%
                } else {
                    0.0
                }
            } else {
                0.15 // Default 15% when no portfolio available
            }
        };

        let diversification_benefit = aggregated_var * diversification_factor;
        let final_var = aggregated_var - diversification_benefit;

        // ES ratio depends on confidence level (1.25x for 95%, 1.15x for 99%)
        let es_ratio = 1.25;
        let final_es = final_var * es_ratio;

        Ok(K2KVaRAggregationResponse {
            correlation_id: msg.correlation_id,
            complete,
            final_var_fp: to_currency_fp(final_var),
            final_es_fp: to_currency_fp(final_es),
            diversification_benefit_fp: to_currency_fp(diversification_benefit),
        })
    }
}

#[async_trait]
impl BatchKernel<MonteCarloVaRInput, MonteCarloVaROutput> for MonteCarloVaR {
    async fn execute(&self, input: MonteCarloVaRInput) -> Result<MonteCarloVaROutput> {
        let start = Instant::now();
        let result = Self::compute(&input.portfolio, input.params);
        Ok(MonteCarloVaROutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// Portfolio Risk Aggregation Kernel
// ============================================================================

/// Portfolio risk aggregation kernel.
///
/// Aggregates risk measures across portfolio with correlation adjustment.
#[derive(Debug, Clone)]
pub struct PortfolioRiskAggregation {
    metadata: KernelMetadata,
}

impl Default for PortfolioRiskAggregation {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioRiskAggregation {
    /// Create a new portfolio risk aggregation kernel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: KernelMetadata::ring("risk/portfolio-aggregation", Domain::RiskAnalytics)
                .with_description("Correlation-adjusted portfolio risk")
                .with_throughput(10_000)
                .with_latency_us(500.0),
        }
    }

    /// Aggregate portfolio risk.
    ///
    /// # Arguments
    /// * `portfolio` - Portfolio to analyze
    /// * `confidence_level` - Confidence level for VaR
    /// * `holding_period` - Holding period in days
    pub fn compute(
        portfolio: &Portfolio,
        confidence_level: f64,
        holding_period: u32,
    ) -> PortfolioRiskResult {
        let n = portfolio.n_assets();
        if n == 0 {
            return PortfolioRiskResult {
                portfolio_var: 0.0,
                portfolio_es: 0.0,
                undiversified_var: 0.0,
                diversification_benefit: 0.0,
                asset_vars: Vec::new(),
                risk_contributions: Vec::new(),
                covariance_matrix: Vec::new(),
            };
        }

        let z = Self::norm_inv(confidence_level);
        let holding_factor = (holding_period as f64).sqrt() / (252.0_f64).sqrt();

        // Calculate covariance matrix
        let mut cov_matrix = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                cov_matrix[i * n + j] = portfolio.volatilities[i]
                    * portfolio.volatilities[j]
                    * portfolio.correlation(i, j);
            }
        }

        // Calculate individual asset VaRs
        let asset_vars: Vec<f64> = portfolio
            .values
            .iter()
            .zip(portfolio.volatilities.iter())
            .map(|(&value, &vol)| value * z * vol * holding_factor)
            .collect();

        // Undiversified VaR (sum of individual VaRs)
        let undiversified_var: f64 = asset_vars.iter().sum();

        // Portfolio variance
        let weights = portfolio.weights();
        let mut portfolio_variance = 0.0;
        for i in 0..n {
            for j in 0..n {
                portfolio_variance += weights[i] * weights[j] * cov_matrix[i * n + j];
            }
        }

        // Portfolio VaR
        let portfolio_var =
            portfolio.total_value() * z * portfolio_variance.sqrt() * holding_factor;

        // Diversification benefit
        let diversification_benefit = undiversified_var - portfolio_var;

        // Risk contributions (Euler allocation)
        let port_vol = portfolio_variance.sqrt();
        let risk_contributions: Vec<f64> = if port_vol > 1e-10 {
            (0..n)
                .map(|i| {
                    let mut cov_i_p = 0.0;
                    for j in 0..n {
                        cov_i_p += weights[j] * cov_matrix[i * n + j];
                    }
                    weights[i] * cov_i_p / port_vol * z * holding_factor * portfolio.total_value()
                })
                .collect()
        } else {
            vec![0.0; n]
        };

        // Expected Shortfall (using normal approximation)
        let pdf_at_z = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let portfolio_es = portfolio.total_value() * port_vol * holding_factor * pdf_at_z
            / (1.0 - confidence_level);

        PortfolioRiskResult {
            portfolio_var,
            portfolio_es,
            undiversified_var,
            diversification_benefit,
            asset_vars,
            risk_contributions,
            covariance_matrix: cov_matrix,
        }
    }

    /// Standard normal inverse CDF.
    fn norm_inv(p: f64) -> f64 {
        MonteCarloVaR::norm_inv(p)
    }
}

impl GpuKernel for PortfolioRiskAggregation {
    fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }
}

#[async_trait]
impl BatchKernel<PortfolioRiskAggregationInput, PortfolioRiskAggregationOutput>
    for PortfolioRiskAggregation
{
    async fn execute(
        &self,
        input: PortfolioRiskAggregationInput,
    ) -> Result<PortfolioRiskAggregationOutput> {
        let start = Instant::now();
        let result = Self::compute(
            &input.portfolio,
            input.confidence_level,
            input.holding_period,
        );
        Ok(PortfolioRiskAggregationOutput {
            result,
            compute_time_us: start.elapsed().as_micros() as u64,
        })
    }
}

// ============================================================================
// Simple RNG for Monte Carlo
// ============================================================================

/// Simple pseudo-random number generator (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate standard normal using Box-Muller transform.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_portfolio() -> Portfolio {
        // Two-asset portfolio with known properties
        Portfolio::new(
            vec![1, 2],
            vec![100_000.0, 100_000.0],
            vec![0.08, 0.10],         // 8%, 10% expected returns
            vec![0.15, 0.20],         // 15%, 20% volatility
            vec![1.0, 0.5, 0.5, 1.0], // Correlation matrix
        )
    }

    fn create_diversified_portfolio() -> Portfolio {
        // Four-asset portfolio with low correlations
        Portfolio::new(
            vec![1, 2, 3, 4],
            vec![50_000.0, 30_000.0, 15_000.0, 5_000.0],
            vec![0.06, 0.08, 0.10, 0.12],
            vec![0.10, 0.15, 0.20, 0.25],
            vec![
                1.0, 0.2, 0.1, 0.0, 0.2, 1.0, 0.3, 0.1, 0.1, 0.3, 1.0, 0.2, 0.0, 0.1, 0.2, 1.0,
            ],
        )
    }

    #[test]
    fn test_monte_carlo_var_metadata() {
        let kernel = MonteCarloVaR::new();
        assert_eq!(kernel.metadata().id, "risk/monte-carlo-var");
        assert_eq!(kernel.metadata().domain, Domain::RiskAnalytics);
    }

    #[test]
    fn test_monte_carlo_var_calculation() {
        let portfolio = create_simple_portfolio();
        let params = VaRParams::new(0.95, 10, 10_000);

        let result = MonteCarloVaR::compute(&portfolio, params);

        assert!(result.var > 0.0, "VaR should be positive");
        assert!(result.expected_shortfall >= result.var, "ES >= VaR");
        assert_eq!(result.confidence_level, 0.95);
        assert_eq!(result.holding_period, 10);

        // VaR should be reasonable (not more than 50% of portfolio)
        assert!(
            result.var < 100_000.0,
            "VaR seems too large: {}",
            result.var
        );
    }

    #[test]
    fn test_var_increases_with_holding_period() {
        let portfolio = create_simple_portfolio();

        let var_1d = MonteCarloVaR::compute(&portfolio, VaRParams::new(0.95, 1, 10_000));
        let var_10d = MonteCarloVaR::compute(&portfolio, VaRParams::new(0.95, 10, 10_000));

        // 10-day VaR should be roughly sqrt(10) times 1-day VaR
        let ratio = var_10d.var / var_1d.var;
        assert!(
            ratio > 2.5 && ratio < 4.0,
            "VaR scaling ratio should be ~sqrt(10): {}",
            ratio
        );
    }

    #[test]
    fn test_var_increases_with_confidence() {
        let portfolio = create_simple_portfolio();

        let var_95 = MonteCarloVaR::compute(&portfolio, VaRParams::new(0.95, 10, 10_000));
        let var_99 = MonteCarloVaR::compute(&portfolio, VaRParams::new(0.99, 10, 10_000));

        assert!(
            var_99.var > var_95.var,
            "99% VaR should exceed 95% VaR: {} vs {}",
            var_99.var,
            var_95.var
        );
    }

    #[test]
    fn test_component_var_sums_to_total() {
        let portfolio = create_diversified_portfolio();
        let params = VaRParams::new(0.95, 10, 10_000);

        let result = MonteCarloVaR::compute(&portfolio, params);

        // Component VaRs should roughly sum to total VaR
        let component_sum: f64 = result.component_var.iter().sum();
        let diff = (component_sum - result.var).abs() / result.var;

        assert!(
            diff < 0.20, // Allow 20% tolerance due to Monte Carlo noise
            "Component VaR sum should be close to total: {} vs {}",
            component_sum,
            result.var
        );
    }

    #[test]
    fn test_portfolio_aggregation_metadata() {
        let kernel = PortfolioRiskAggregation::new();
        assert_eq!(kernel.metadata().id, "risk/portfolio-aggregation");
    }

    #[test]
    fn test_portfolio_aggregation() {
        let portfolio = create_diversified_portfolio();
        let result = PortfolioRiskAggregation::compute(&portfolio, 0.95, 10);

        assert!(result.portfolio_var > 0.0);
        assert!(result.undiversified_var > result.portfolio_var);
        assert!(result.diversification_benefit > 0.0);
    }

    #[test]
    fn test_diversification_benefit() {
        let portfolio = create_diversified_portfolio();
        let result = PortfolioRiskAggregation::compute(&portfolio, 0.95, 10);

        // Diversification benefit = undiversified - portfolio
        assert!(
            (result.diversification_benefit - (result.undiversified_var - result.portfolio_var))
                .abs()
                < 1.0
        );

        // With low correlations, should have significant benefit
        let benefit_pct = result.diversification_benefit / result.undiversified_var;
        assert!(
            benefit_pct > 0.10,
            "Should have >10% diversification benefit: {}%",
            benefit_pct * 100.0
        );
    }

    #[test]
    fn test_risk_contributions_sum() {
        let portfolio = create_diversified_portfolio();
        let result = PortfolioRiskAggregation::compute(&portfolio, 0.95, 10);

        let contrib_sum: f64 = result.risk_contributions.iter().sum();

        // Risk contributions should sum to portfolio VaR
        let diff = (contrib_sum - result.portfolio_var).abs() / result.portfolio_var;
        assert!(
            diff < 0.01,
            "Risk contributions should sum to portfolio VaR: {} vs {}",
            contrib_sum,
            result.portfolio_var
        );
    }

    #[test]
    fn test_empty_portfolio() {
        let empty = Portfolio::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

        let var_result = MonteCarloVaR::compute(&empty, VaRParams::default());
        assert_eq!(var_result.var, 0.0);

        let agg_result = PortfolioRiskAggregation::compute(&empty, 0.95, 10);
        assert_eq!(agg_result.portfolio_var, 0.0);
    }

    #[test]
    fn test_covariance_matrix() {
        let portfolio = create_simple_portfolio();
        let result = PortfolioRiskAggregation::compute(&portfolio, 0.95, 10);

        // Covariance matrix should be symmetric
        assert_eq!(result.covariance_matrix.len(), 4);
        assert!((result.covariance_matrix[1] - result.covariance_matrix[2]).abs() < 1e-10);

        // Diagonal should be variance
        let var1 = 0.15 * 0.15;
        let var2 = 0.20 * 0.20;
        assert!((result.covariance_matrix[0] - var1).abs() < 1e-10);
        assert!((result.covariance_matrix[3] - var2).abs() < 1e-10);
    }
}
