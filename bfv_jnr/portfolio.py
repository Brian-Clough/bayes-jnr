# bfv_jnr/portfolio.py
import logging
import numpy as np
from typing import Dict, List, Any
from bfv_jnr.engine import JNRScaleReconciliationEngine

logger = logging.getLogger("Bayes-JNR")

class JNRJurisdictionalLedgerRegistry:
    """
    Sovereign Registry Simulator that programmatically coordinates a portfolio 
    of multiple nested REDD+ projects under a macro Forest Reference Level (FRL) Prior ceiling.
    """
    def __init__(self, sovereign_country_name: str, macro_frl_mean: float, macro_frl_sd: float, total_jurisdictional_ha: float):
        if macro_frl_mean <= 0 or macro_frl_sd <= 0 or total_jurisdictional_ha <= 0:
            raise ValueError("All jurisdictional parameters must be strictly positive scalars.")
            
        self.country = str(sovereign_country_name)
        self.macro_mean = float(macro_frl_mean)
        self.macro_sd = float(macro_frl_sd)
        self.total_area = float(total_jurisdictional_ha)
        
        # Absolute carbon asset volume ceiling for the sovereign nation (t C)
        self.total_sovereign_pool_tC = self.macro_mean * self.total_area
        
        # Internal state to track registered project nodes
        self.nested_projects: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Initialized Sovereign Registry Matrix for {self.country}. Cap: {self.total_sovereign_pool_tC:,.1f} t C")

    def register_nested_project(self, project_id: str, area_ha: float, 
                                observed_mean: float, standard_error: float, n_plots: int) -> None:
        """Enrolls a unique project boundary node into the jurisdictional tracking ledger."""
        if area_ha <= 0 or observed_mean <= 0 or standard_error <= 0 or n_plots < 2:
            raise ValueError(f"Invalid biometrics payload for project enrollment: {project_id}")
            
        self.nested_projects[project_id] = {
            "area_ha": float(area_ha),
            "project_mean": float(observed_mean),
            "project_se": float(standard_error),
            "n_plots": int(n_plots)
        }
        logger.info(f"Successfully enrolled Nested Project Node [{project_id}] — Footprint: {area_ha:,} Ha")

    def process_portfolio_issuance(self) -> Dict[str, Any]:
        """
        Executes parallel Bayesian scale inversions across all enrolled projects,
        sums total solvent credit allocations, and calculates the residual carbon pool
        available for Sovereign Jurisdictional Credit issuance under the absolute NDC ceiling.
        """
        if not self.nested_projects:
            raise RuntimeError("Cannot process issuance ledger on an empty jurisdictional portfolio.")

        portfolio_results = {}
        total_project_area_committed = 0.0
        total_project_solvent_issuance_tC = 0.0
        
        # Initialize a base scale engine matching the national prior profile
        inversion_engine = JNRScaleReconciliationEngine(jurisdictional_mean=self.macro_mean, jurisdictional_sd=self.macro_sd)
        
        # ─── STEP 1: RESOLVE NESTED PROJECT POOLS VIA MCMC ────────────────────
        for p_id, p_data in self.nested_projects.items():
            area = p_data["area_ha"]
            total_project_area_committed += area
            
            # Execute MCMC simulation loop for the project node
            verdict = inversion_engine.reconcile_project_metrics(
                project_mean=p_data["project_mean"],
                project_se=p_data["project_se"],
                n_plots=p_data["n_plots"],
                method="simulation"
            )
            
            # Compute absolute solvent volume cap for this specific node
            project_cap_tC = verdict["solvent_credit_issuance_cap"] * area
            total_project_solvent_issuance_tC += project_cap_tC
            
            portfolio_results[p_id] = {
                "hectares_monitored": area,
                "reconciled_mean_tC_ha": verdict["posterior_mean_tC_ha"],
                "solvent_volume_issued_tC": round(project_cap_tC, 1)
            }

        # ─── STEP 2: CALCULATE JURISDICTIONAL RESIDUAL ASSETS ─────────────────
        if total_project_area_committed > self.total_area:
            raise ValueError("Sovereign Land Breach: Nested project boundaries exceed national geographic limits.")

        residual_jurisdictional_area_ha = self.total_area - total_project_area_committed
        
        # The leftover baseline volume strictly allocated to non-nested state forests
        expected_residual_pool_tC = residual_jurisdictional_area_ha * self.macro_mean
        
        # Sovereign Buffer Factor: Ensure total project issuance + state allocation fits under the hard pool cap
        total_ledger_claimed_tC = total_project_solvent_issuance_tC + expected_residual_pool_tC
        
        # Calculate final national solvency margin
        is_ndc_compliant = total_ledger_claimed_tC <= self.total_sovereign_pool_tC
        sovereign_balance_delta_tC = self.total_sovereign_pool_tC - total_project_solvent_issuance_tC
        
        return {
            "sovereign_ledger_metadata": {
                "country": self.country,
                "absolute_ndc_volume_ceiling_tC": round(self.total_sovereign_pool_tC, 1),
                "total_geographic_area_ha": self.total_area
            },
            "nested_project_issuance_ledger": portfolio_results,
            "portfolio_aggregates": {
                "total_nested_area_ha": total_project_area_committed,
                "total_nested_credits_issued_tC": round(total_project_solvent_issuance_tC, 1)
            },
            "jurisdictional_residual_pool": {
                "unallocated_state_forest_area_ha": residual_jurisdictional_area_ha,
                "maximum_sovereign_issuance_pool_tC": round(max(0.0, sovereign_balance_delta_tC), 1),
                "ndc_accounting_compliance_solvent": is_ndc_compliant
            }
        }
