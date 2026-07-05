# bayes-jnr: Hierarchical Bayesian Inversion for Nested Carbon Frameworks

`bayes-jnr` is an open-source Python library engineered to resolve the structural scale asymmetries trapped within Jurisdictional and Nested REDD+ (JNR) frameworks and standardized compliance baselines (e.g., Verra VM0048).

## The Engineering Problem
Current voluntary and compliance registries attempt to reconcile top-down regional Forest Reference Levels (FRL) with bottom-up project field inventories using rigid, manual, pro-rata allocation tables. This compliance math creates a severe data-reconciliation bottleneck: it either dilutes high-performance project interventions or completely masks structural baseline errors at the boundary lines. 

By treating the top-down jurisdictional FRL as a fixed, deterministic constant (Empirical Bayes / Closed-Form), current models systematically underestimate the heavy-tail variance of carbon assets, exposing the market to future over-crediting audits and greenwashing vulnerabilities.

## The Architectural Fix
`bayes-jnr` replaces document compliance with an analytical software engine. By modeling the top-down jurisdictional FRL as a structural **Prior probability distribution** and the bottom-up project field cruise as an independent **Observational likelihood layer (Student-t)**, the package executes an independent Metropolis-Hastings Markov Chain Monte Carlo (MCMC) simulation loop. 

The math programmatically penalizes small sample networks and integrates nested parameter uncertainties non-linearly, self-correcting for true biophysical risk without requiring bureaucratic human intervention.

## Installation
```bash
pip install git+https://github.com
```

## Quick Start Pipeline
```python
from bfv_jnr import JNRScaleReconciliationEngine

# Initialize with regional JNR Baseline (FRL Mean = 145.0 t C/ha, Spatial SD = 18.5)
engine = JNRScaleReconciliationEngine(jurisdictional_mean=145.0, jurisdictional_sd=18.5)

# Evaluate a local 10-plot sample inventory network (Mean = 162.0, SE = 4.5)
# Defaults natively to full MCMC simulation modeling
verdict = engine.reconcile_project_metrics(
    project_mean=162.0, 
    project_se=4.5, 
    n_plots=10, 
    method="simulation"
)

print(f"Risk-Adjusted Posterior Mean: {verdict['posterior_mean_tC_ha']} t C/ha")
print(f"95% Solvent Credit Issuance Cap: {verdict['solvent_credit_issuance_cap']} t C/ha")
```

## Benchmarking Risk Underestimation
The library contains a built-in benchmark toggle allowing users to compare full joint simulations against legacy closed-form analytical models. Run the automated test suite to see the exact mathematical overcrediting delta created by fixing prior hyperparameters:
```bash
python -m unittest tests/test_library.py
```

## Core Pedigree
Asymptotic variance scaling, spatial covariance matrices, and multi-layered error propagation models heavily informed by Clark et al. (*Ecological Applications*), Chave et al. (*Global Change Biology*), and Dietze (*Ecological Forecasting*).
