# tests/test_core.py
import unittest
from bfv_jnr.engine import JNRScaleReconciliationEngine
from bfv_jnr.risk_engine import JNRBoundedRiskEngine
from bfv_jnr.finance import DMRVFinancialTranslationEngine

class TestJNRCoreMathematicalEngine(unittest.TestCase):
    def setUp(self):
        """Set up standard unbounded, bounded, and financial engines."""
        # Unbounded Engine Prior: 145.0 ± 18.5 t C/ha
        self.carbon_engine = JNRScaleReconciliationEngine(jurisdictional_mean=145.0, jurisdictional_sd=18.5)
        # Bounded Risk Engine Prior: alpha=4, beta=6 (40% expected harvest risk)
        self.risk_engine = JNRBoundedRiskEngine(prior_alpha=4.0, prior_beta=6.0)
        # Financial Engine: $18.50 per ton spot price
        self.finance_engine = DMRVFinancialTranslationEngine(asset_spot_price_usd=18.50)

    def test_unbounded_mcmc_vs_analytical_uncertainty(self):
        """Verify MCMC and analytical paths execute and simulation expands tail risk."""
        params = {"project_mean": 162.0, "project_se": 4.5, "n_plots": 10}
        
        sim_res = self.carbon_engine.reconcile_project_metrics(**params, method="simulation")
        ana_res = self.carbon_engine.reconcile_project_metrics(**params, method="analytical")
        
        # Verify basic structural keys exist
        self.assertEqual(sim_results["method_executed"] if 'sim_results' in locals() else sim_res["method_executed"], "simulation_mcmc")
        self.assertEqual(ana_results["method_executed"] if 'ana_results' in locals() else ana_res["method_executed"], "analytical_conjugate")
        
        # PROOF: Full joint simulation must expose broader posterior uncertainty bounds
        # than closed-form analytical assumptions that treat prior parameters as constants
        self.assertGreaterEqual(sim_res["posterior_se_tC_ha"], ana_res["posterior_se_tC_ha"])

    def test_bounded_risk_mcmc_boundary_walls(self):
        """Verify that the bounded MCMC risk engine respects strict 0-1 probability limits."""
        # 50 total pixel trials, 12 observed harvest disturbances
        risk_res = self.risk_engine.reconcile_harvest_risk(local_observed_events=12, local_total_trials=50)
        
        # Expected risk should shift toward local empirical fraction (12/50 = 24%) from prior (40%)
        self.assertLess(risk_res["posterior_expected_risk_fraction"], 0.35)
        # Conservative risk ceiling (95th percentile) must safely buffer the expected risk fraction
        self.assertGreater(risk_res["conservative_risk_ceiling_95"], risk_res["posterior_expected_risk_fraction"])
        # Probability parameter must never leak outside physical bounds
        self.assertLess(risk_res["conservative_risk_ceiling_95"], 1.0)
        self.assertGreater(risk_res["posterior_expected_risk_fraction"], 0.0)

    def test_financial_risk_translation_and_var(self):
        """Verify that biophysical uncertainty maps accurately to financial Value-at-Risk (VaR)."""
        mock_mcmc_input = {
            "posterior_mean_tC_ha": 158.42,
            "posterior_se_tC_ha": 5.02,
            "solvent_credit_issuance_cap": 148.1
        }
        
        prospectus = self.finance_engine.compute_issuance_solvency(mock_mcmc_input, total_project_hectares=10000.0)
        
        # Verify monetary structures
        valuation = prospectus["portfolio_valuation"]
        risk_exp = prospectus["institutional_risk_exposure"]
        
        # Underwritten solvent valuation must always be lower than gross expected value to protect capital
        self.assertLess(valuation["solvent_underwritten_valuation_usd"], valuation["expected_gross_valuation_usd"])
        # 99% VaR must be structurally higher than 95% VaR as the loss distribution deepens
        self.assertGreater(risk_exp["financial_value_at_risk_99_pct_usd"], risk_exp["financial_value_at_risk_95_pct_usd"])
        # Solvency ratio must remain bounded beneath 1.0
        self.assertLess(risk_exp["issuance_solvency_ratio"], 1.0)

    def test_illegal_input_safeguards(self):
        """Ensure all core modules actively reject physically impossible dimensions."""
        with self.assertRaises(ValueError):
            JNRScaleReconciliationEngine(jurisdictional_mean=-100, jurisdictional_sd=10)
        with self.assertRaises(ValueError):
            JNRBoundedRiskEngine(prior_alpha=5, prior_beta=-2)
        with self.assertRaises(ValueError):
            DMRVFinancialTranslationEngine(asset_spot_price_usd=-5.00)

if __name__ == "__main__":
    unittest.main()
