# bfv_jnr/risk_engine.py
import logging
import numpy as np
import scipy.stats as stats
from typing import Dict, Any

# Configure structured logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Bayes-JNR")

class JNRBoundedRiskEngine:
    """
    Advanced Hierarchical Bayesian Bounded Risk Engine executing an MCMC 
    Metropolis-Hastings simulation loop to reconcile bounded [0,1] JNR risk maps 
    with empirical local observations under joint uncertainty.
    """
    def __init__(self, prior_alpha: float, prior_beta: float):
        """
        Initialize using shape parameters derived from a regional JNR risk map.
        Prior Mean Risk = alpha / (alpha + beta)
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Prior alpha and beta shape parameters must be positive scalars.")
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def _ln_joint_posterior_density(self, p_risk: float, local_events: int, total_trials: int) -> float:
        """
        Computes the log un-normalized joint posterior density for a probability parameter.
        Enforces a strict [0, 1] boundary wall for physical solvency.
        """
        # Strict boundary guard: Risk fractions outside [0, 1] carry zero probability
        if p_risk <= 0.0 or p_risk >= 1.0:
            return -np.inf
            
        # Prior Log-Density: P(p_risk | Regional Map Shape Parameters)
        ln_prior = stats.beta.logpdf(p_risk, self.prior_alpha, self.prior_beta)
        
        # Likelihood Log-Density: P(Local Observations | p_risk)
        ln_likelihood = stats.binom.logpmf(local_events, n=total_trials, p=p_risk)
        
        return ln_prior + ln_likelihood

    def reconcile_harvest_risk(self, local_observed_events: int, local_total_trials: int,
                               n_samples: int = 50000, burn_in: int = 10000) -> Dict[str, Any]:
        """
        Simulates the true joint risk posterior distribution via Metropolis-Hastings MCMC.
        Bypasses Empirical Beta limitations to map conservative 95% risk ceilings.
        """
        if local_observed_events < 0 or local_total_trials <= 0:
            raise ValueError("Observation metrics must be non-negative with trials > 0.")
        if local_observed_events > local_total_trials:
            raise ValueError("Observed risk events cannot exceed total evaluated pixel trials.")

        logger.info(f"Executing Bounded MCMC risk simulation ({n_samples} iterations)...")
        trace = np.zeros(n_samples)
        
        # Initialize the Markov Chain at the empirical local sample fraction
        current_p = local_observed_events / local_total_trials
        # Safeguard initialization from boundary edges
        current_p = max(0.01, min(0.99, current_p))
        
        current_ln_post = self._ln_joint_posterior_density(current_p, local_observed_events, local_total_trials)
        
        # Propose jumps using a tight variance window suited for [0, 1] space
        proposal_sd = 0.05
        accepted = 0
        
        # --- THE BOUNDED MCMC RISK LOOP ---
        for i in range(n_samples):
            # Propose a candidate risk fraction via a normal random walk
            proposed_p = np.random.normal(current_p, proposal_sd)
            proposed_ln_post = self._ln_joint_posterior_density(proposed_p, local_observed_events, local_total_trials)
            
            # Evaluate the Metropolis-Hastings acceptance ratio
            if np.log(np.random.uniform(0, 1)) < (proposed_ln_post - current_ln_post):
                current_p = proposed_p
                current_ln_post = proposed_ln_post
                if i >= burn_in:
                    accepted += 1
            trace[i] = current_p
            
        # Strip out the non-stationary burn-in phase
        post_burn_trace = trace[burn_in:]
        
        posterior_mean = np.mean(post_burn_trace)
        posterior_sd = np.std(post_burn_trace)
        
        # Compute the explicit conservative 95th percentile risk ceiling
        # For risk and deforestation, the upper bound is our solvent defense margin
        risk_ceiling_95 = np.percentile(post_burn_trace, 95.0)
        
        return {
            "method_executed": "simulation_bounded_mcmc",
            "posterior_expected_risk_fraction": round(posterior_mean, 4),
            "posterior_risk_uncertainty_sd": round(posterior_sd, 4),
            "conservative_risk_ceiling_95": round(risk_ceiling_95, 4),
            "mcmc_acceptance_rate": round(accepted / (n_samples - burn_in), 3),
            "simulated_samples_retained": len(post_burn_trace)
        }
