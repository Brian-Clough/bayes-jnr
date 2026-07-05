# bfv_jnr/engine.py
import logging
import numpy as np
import scipy.stats as stats
from typing import Dict, Any

# Configure clean, non-anxious logging pipelines for cloud execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Bayes-JNR")

class JNRScaleReconciliationEngine:
    """
    Advanced Hierarchical Bayesian Scale Reconciliation Engine. Integrates 
    coarse, top-down Jurisdictional Forest Reference Levels (Priors) with 
    bottom-up project field inventories (Likelihoods), offering both closed-form
    analytical updates and full MCMC simulation paths.
    """
    def __init__(self, jurisdictional_mean: float, jurisdictional_sd: float):
        if jurisdictional_mean <= 0 or jurisdictional_sd <= 0:
            raise ValueError("Jurisdictional parameters must be positive scalars.")
        self.jur_mean = float(jurisdictional_mean)
        self.jur_sd = float(jurisdictional_sd)
        self.prior_precision = 1.0 / (self.jur_sd ** 2)

    def _ln_joint_posterior_density(self, theta: float, p_mean: float, p_se: float, df: int) -> float:
        """
        Computes un-normalized joint log posterior using a heavy-tailed 
        Student-t project likelihood to natively penalize small plot networks.
        """
        ln_prior = stats.norm.logpdf(theta, loc=self.jur_mean, scale=self.jur_sd)
        ln_likelihood = stats.t.logpdf(p_mean, df=df, loc=theta, scale=p_se)
        return ln_prior + ln_likelihood

    def reconcile_project_metrics(self, project_mean: float, project_se: float, n_plots: int, 
                                  method: str = "simulation", n_samples: int = 50000, 
                                  burn_in: int = 10000) -> Dict[str, Any]:
        """
        Reconciles top-down and bottom-up data vectors. 
        Toggles between 'simulation' (Full MCMC) and 'analytical' (Closed-form conjugate).
        """
        if project_mean <= 0 or project_se <= 0 or n_plots < 2:
            raise ValueError("Invalid project metrics or insufficient plot sample size.")
        
        if method not in ["simulation", "analytical"]:
            raise ValueError("Method must be either 'simulation' or 'analytical'.")

        df = n_plots - 1
        p_mean = float(project_mean)
        p_se = float(project_se)

        # ─── METHOD 1: THE FULL MCMC SIMULATION RUNNER (DEFAULT) ────────────────
        if method == "simulation":
            logger.info(f"Executing Markov Chain Monte Carlo simulation ({n_samples} iterations)...")
            trace = np.zeros(n_samples)
            current_theta = p_mean
            current_ln_post = self._ln_joint_posterior_density(current_theta, p_mean, p_se, df)
            
            proposal_sd = p_se * 1.2
            accepted = 0
            
            for i in range(n_samples):
                proposed_theta = np.random.normal(current_theta, proposal_sd)
                proposed_ln_post = self._ln_joint_posterior_density(proposed_theta, p_mean, p_se, df)
                
                # Evaluate Metropolis-Hastings acceptance criteria
                if np.log(np.random.uniform(0, 1)) < (proposed_ln_post - current_ln_post):
                    current_theta = proposed_theta
                    current_ln_post = proposed_ln_post
                    if i >= burn_in:
                        accepted += 1
                trace[i] = current_theta
                
            post_burn_trace = trace[burn_in:]
            post_mean = np.mean(post_burn_trace)
            post_sd = np.std(post_burn_trace)
            issuance_cap = np.percentile(post_burn_trace, 2.5)
            
            return {
                "method_executed": "simulation_mcmc",
                "posterior_mean_tC_ha": round(post_mean, 2),
                "posterior_se_tC_ha": round(post_sd, 2),
                "solvent_credit_issuance_cap": round(max(0.0, issuance_cap), 1),
                "mcmc_acceptance_rate": round(accepted / (n_samples - burn_in), 3),
                "degrees_of_freedom": df,
                "simulated_samples_retained": len(post_burn_trace)
            }

        # ─── METHOD 2: THE LEGACY ANALYTICAL BENCHMARK TOGGLE ───────────────────
        elif method == "analytical":
            logger.info("Executing closed-form conjugate normal scale update...")
            # Apply basic small-sample critical value inflating penalty
            t_crit, z_crit = stats.t.ppf(0.95, df), stats.norm.ppf(0.95)
            adjusted_se = p_se * (t_crit / z_crit)
            project_precision = 1.0 / (adjusted_se ** 2)
            
            posterior_precision = self.prior_precision + project_precision
            posterior_variance = 1.0 / posterior_precision
            post_sd = np.sqrt(posterior_variance)
            
            post_mean = posterior_variance * (
                (self.jur_mean * self.prior_precision) + (p_mean * project_precision)
            )
            issuance_cap = stats.norm.ppf(0.025, loc=post_mean, scale=post_sd)
            
            return {
                "method_executed": "analytical_conjugate",
                "posterior_mean_tC_ha": round(post_mean, 2),
                "posterior_se_tC_ha": round(post_sd, 2),
                "solvent_credit_issuance_cap": round(max(0.0, issuance_cap), 1),
                "data_fidelity_weight_fraction": round(project_precision / posterior_precision, 4),
                "degrees_of_freedom": df
            }
