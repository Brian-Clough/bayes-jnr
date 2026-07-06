# bayes-jnr: Hierarchical Bayesian Inversion for Sovereign JNR Registries

`bayes-jnr` is an institutional-grade open-source Python library engineered to resolve the structural scale asymmetries and risk underestimation errors trapped within Jurisdictional and Nested REDD+ (JNR) frameworks and compliance baselines (e.g., Verra VM0048).

## The Engineering Problem
Current voluntary and compliance registries attempt to reconcile top-down regional Forest Reference Levels (FRL) with bottom-up project field inventories using rigid, manual, pro-rata administrative spreadsheets. This compliance math creates a severe data-reconciliation bottleneck: it either dilutes high-performance project interventions or completely masks structural baseline errors at the boundary lines. 

By treating top-down regional map variables (biomass means, harvest risk ratios, deforestation map shapes) as fixed, deterministic constants, current closed-form analytical models systematically underestimate the heavy-tail variance of nature-based assets. This exposes institutional investors, insurers, and host-country NDC ledgers to massive over-crediting liabilities and greenwashing audits.

## The Architectural Solution
`bayes-jnr` replaces administrative document compliance with a code-driven risk underwriting suite. The package splits scaling parameters by their physical boundaries and simulates true joint probability distributions using native Metropolis-Hastings Markov Chain Monte Carlo (MCMC) random walks:

1. **Unbounded Asset Inversion (`engine.py`):** Models unbounded forest biomass densities (t C/ha) using heavy-tailed Student-t Likelihoods nested under normal regional Priors, programmatically penalizing sparse, high-variance plot networks based on sample size.
2. **Bounded Risk Simulation (`risk_engine.py`):** Models strictly 0-1 bounded risk layers (harvest maps, deforestation vectors, canopy cover fractions) via a bounded MCMC random walk with strict physical boundary reflection walls (≤ 0 or ≥ 1).
3. **Sovereign Portfolio Management (`portfolio.py`):** Co-ordinates multi-project nested issuance concurrently under an absolute macro NDC volume ceiling, automatically accounting for unallocated state-forest residuals.
4. **Financial Risk Translation (`finance.py`):** Converts biophysical MCMC traces into Wall-Street-grade underwriting parameters—calculating Asset Value-at-Risk (VaR) and project Issuance Solvency Ratios.

## Repository Architecture
```text
bayes-jnr/
├── bfv_jnr/
│   ├── __init__.py      # Package routing optimization
│   ├── engine.py        # Unbounded biomass MCMC engine
│   ├── risk_engine.py   # Bounded harvest/deforestation MCMC engine
│   ├── prep.py          # Top-down JNR allocation workflows
│   ├── finance.py       # Financial Risk & Value-at-Risk translation
│   └── portfolio.py     # Sovereign Portfolio Registry ledger layer
└── tests/
    ├── test_core.py     # Core mathematical validation checks
    └── test_portfolio.py# Multi-project constraint audits
```

## Installation
```bash
pip install git+https://github.com/Brian-Clough/bayes-jnr
```

## End-to-End Compliance Pipeline
```python
from bfv_jnr.portfolio import JNRJurisdictionalLedgerRegistry

# 1. Initialize a sovereign territory (e.g., Mean Prior = 150 t C/ha, Spatial SD = 15.0)
gabon_registry = JNRJurisdictionalLedgerRegistry(
    sovereign_country_name="Gabon Republic",
    macro_frl_mean=150.0,
    macro_frl_sd=15.0,
    total_jurisdictional_ha=2500000.0
)

# 2. Enroll a high-precision project (Dense plots, low Standard Error)
gabon_registry.register_nested_project(
    project_id="PROJ-HIGH-FIDELITY", area_ha=150000.0, observed_mean=165.0, standard_error=2.5, n_plots=80
)

# 3. Enroll a low-precision project (Sparse plots, high Standard Error)
gabon_registry.register_nested_project(
    project_id="PROJ-LOW-FIDELITY", area_ha=200000.0, observed_mean=172.0, standard_error=16.5, n_plots=12
)

# 4. Execute programmatic portfolio issuance and audit ledger solvency
prospectus = gabon_registry.process_portfolio_issuance()
print(prospectus["jurisdictional_residual_pool"])
```

## Verifying Risk Underestimation Benchmarks
The repository contains a built-in automated test suite that compares full joint MCMC simulations against legacy closed-form analytical models. Run the testing protocols locally to measure the exact mathematical over-crediting delta created by fixing prior hyperparameters:
```bash
python -m unittest discover tests/
```

## Scientific Pedigree
Asymptotic variance scaling, spatial covariance matrices, and multi-layered hierarchical error propagation heavily informed by Clark et al. (*Ecological Applications*), Chave et al. (*Global Change Biology*), and Dietze (*Ecological Forecasting*).

## License
MIT Open Science License.

