# tests/test_portfolio.py
import unittest
from bfv_jnr.portfolio import JNRJurisdictionalLedgerRegistry

class TestJNRPortfolioLedger(unittest.TestCase):
    def setUp(self):
        """Set up a standard sovereign jurisdiction with a known FRL ceiling."""
        # 1 million hectares total, FRL mean of 150.0 t C/ha, SD of 15.0
        self.registry = JNRJurisdictionalLedgerRegistry(
            sovereign_country_name="Test Republic",
            macro_frl_mean=150.0,
            macro_frl_sd=15.0,
            total_jurisdictional_ha=1000000.0
        )

    def test_multi_project_issuance_throttling(self):
        """Verify that high-variance projects are throttled while precise projects are rewarded."""
        # Project 1: High-Fidelity (Dense plots, low SE)
        self.registry.register_nested_project(
            project_id="PROJ-HIGH-FIDELITY",
            area_ha=100000.0,
            observed_mean=165.0,
            standard_error=2.5,
            n_plots=80
        )

        # Project 2: Low-Fidelity (Sparse plots, high SE)
        self.registry.register_nested_project(
            project_id="PROJ-LOW-FIDELITY",
            area_ha=100000.0,
            observed_mean=172.0,
            standard_error=16.5,
            n_plots=12
        )

        prospectus = self.registry.process_portfolio_issuance()
        
        # Verify land area math tracking
        self.assertEqual(prospectus["portfolio_aggregates"]["total_nested_area_ha"], 200000.0)
        self.assertEqual(prospectus["jurisdictional_residual_pool"]["unallocated_state_forest_area_ha"], 800000.0)
        
        # PROOF: Low-fidelity project must have its reconciled mean pulled back heavily
        # toward the prior mean (150) compared to its inflated raw claim (172)
        high_fid_mean = prospectus["nested_project_issuance_ledger"]["PROJ-HIGH-FIDELITY"]["reconciled_mean_tC_ha"]
        low_fid_mean = prospectus["nested_project_issuance_ledger"]["PROJ-LOW-FIDELITY"]["reconciled_mean_tC_ha"]
        
        self.assertGreater(high_fid_mean, 160.0)
        self.assertLess(low_fid_mean, 155.0)  # Gravity drag pulled it back to safety

    def test_sovereign_land_breach_safeguard(self):
        """Ensure the ledger throws an exception if nested claims exceed total country geography."""
        self.registry.register_nested_project(
            project_id="PROJ-BREACH",
            area_ha=1200000.0,  # Exceeds the 1-million-hectare total country limit
            observed_mean=150.0,
            standard_error=5.0,
            n_plots=50
        )
        with self.assertRaises(ValueError):
            self.registry.process_portfolio_issuance()

if __name__ == "__main__":
    unittest.main()
