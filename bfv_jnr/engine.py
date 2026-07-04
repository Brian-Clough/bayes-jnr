# bfv_jnr/engine.py
import numpy as np
import scipy.stats as stats
from typing import Dict, Any

class JNRScaleReconciliationEngine:
    def __init__(self, jurisdictional_mean: float, jurisdictional_sd: float):
        if jurisdictional_mean <= 0 or jurisdictional_sd <= 0:
            raise ValueError("Jurisdictional parameters must be positive.")
        self.jur_mean = float(jurisdictional_mean)
        self.jur_sd = float(jurisdictional_sd)
        self.prior_precision = 1.0 / (self.jur_sd ** 2)

    def reconcile_project_metrics(self, project_mean: float, project_se: float) -> Dict[str, Any]:
        if project_mean <= 0 or project_se <= 0:
            raise ValueError("Project parameters must be positive.")
        p_mean = float(project_mean)
        p_se = float(project_se)
        project_precision = 1.0 / (p_se ** 2)
        
        posterior_precision = self.prior_precision + project_precision
        posterior_variance = 1.0 / posterior_precision
        posterior_sd = np.sqrt(posterior_variance)
        
        posterior_mean = posterior_variance * ((self.jur_mean * self.prior_precision) + (p_mean * project_precision))
        credit_issuance_cap_95 = stats.norm.ppf(0.025, loc=posterior_mean, scale=posterior_sd)
        
        return {
            "posterior_mean_tC_ha": round(posterior_mean, 2),
            "posterior_se_tC_ha": round(posterior_sd, 2),
            "solvent_credit_issuance_cap": round(max(0.0, credit_issuance_cap_95), 1),
            "data_fidelity_weight_fraction": round(project_precision / posterior_precision, 4)
        }
