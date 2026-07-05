# bfv_jnr/prep.py
import logging
from typing import Dict, Any

logger = logging.getLogger("Bayes-JNR")

class JNRWorkflowPreparer:
    """
    Data preparation layer mapped to Jurisdictional and Nested REDD+ (JNR) 
    workflows. Reconciles raw localized project attributes against centralized 
    registry risk maps and macro-allocation ceilings.
    """
    
    @staticmethod
    def calculate_jurisdictional_allocation(
        project_area_ha: float,
        regional_risk_weight: float,
        central_jurisdictional_frl_total: float,
        total_regional_allocated_risk_units: float
    ) -> Dict[str, Any]:
        """
        Executes a top-down JNR allocation workflow. Translates centralized 
        regional risk map attributes into a specific, project-level baseline ceiling.
        """
        if project_area_ha <= 0 or regional_risk_weight <= 0:
            raise ValueError("Project footprints and risk attributes must be positive scalars.")
            
        # Calculate the project's relative share of the macro jurisdiction's risk footprint
        project_risk_share = (project_area_ha * regional_risk_weight)
        allocation_fraction = project_risk_share / total_regional_allocated_risk_units
        
        # Determine the maximum allowable baseline carbon ceiling for this nested asset
        allocated_baseline_ceiling_tC = central_jurisdictional_frl_total * allocation_fraction
        allocated_baseline_mean_tC_ha = allocated_baseline_ceiling_tC / project_area_ha
        
        logger.info(f"JNR Top-Down Allocation mapped. Assigned Baseline: {allocated_baseline_mean_tC_ha:.2f} t C/ha")
        
        return {
            "project_risk_share_units": round(project_risk_share, 2),
            "jurisdictional_allocation_fraction": round(allocation_fraction, 5),
            "allocated_baseline_ceiling_tC": round(allocated_baseline_ceiling_tC, 2),
            "allocated_baseline_mean_tC_ha": round(allocated_baseline_mean_tC_ha, 2)
        }

    @staticmethod
    def reconcile_pro_rata_scaling(
        nested_project_baselines_sum: float,
        official_jurisdictional_frl_ceiling: float,
        local_project_raw_baseline: float
    ) -> Dict[str, Any]:
        """
        Executes the mandatory JNR alignment gate workflow. Applies Verra-tier 
        pro-rata downward adjustment scaling if the aggregate of nested projects 
        exceeds the sovereign national reference ceiling.
        """
        if nested_project_baselines_sum <= 0 or official_jurisdictional_frl_ceiling <= 0:
            raise ValueError("Registry baseline totals must be positive vectors.")
            
        # If projects collectively overestimate, calculate the downward alignment factor
        if nested_project_baselines_sum > official_jurisdictional_frl_ceiling:
            adjustment_factor = official_jurisdictional_frl_ceiling / nested_project_baselines_sum
            compliance_status = "ADJUSTMENT_REQUIRED"
            logger.warning(f"JNR Ceiling Breached. Applying Pro-Rata Scaling Factor: {adjustment_factor:.4f}")
        else:
            adjustment_factor = 1.0
            compliance_status = "COMPLIANT_WITHIN_CEILING"
            
        adjusted_local_baseline = local_project_raw_baseline * adjustment_factor
        
        return {
            "jnr_compliance_status": compliance_status,
            "pro_rata_adjustment_factor": round(adjustment_factor, 4),
            "adjusted_local_baseline_tC_ha": round(adjusted_local_baseline, 2),
            "leakage_buffer_retained_tC_ha": round(local_project_raw_baseline - adjusted_local_baseline, 2)
        }
