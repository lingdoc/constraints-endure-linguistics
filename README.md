# Constraints that Endure: Assessing Model Robustness in Linguistic Typology

This repository contains the replication pipeline and diagnostic code used to analyze model robustness and isolate parameter volatility within global typological databases, using the cross-linguistic universals framework from Verkerk et al. (2026).

## The Core Problem: Isolate Variance Traps

The original study utilized a Bayesian spatiophylogenetic mixed model (`brms` in R) across 100 posterior phylogenetic trees to evaluate support for 191 binary grammatical universals. However, on highly fragmented typological networks—where structural density drops below a critical floor (𝜅_structural ≤ 0.10)—the ancestral trees shatter into disconnected family islands.

When a multi-level sampling algorithm encounters these historical isolates (single language families with no close tree relatives), the parameter space faces severe numerical stress:
* **Parameter Collapse:** Unanchored localized signals are over-smoothed and flatline toward global intercept means.
* **Parameter Explosion:** Runaway variance estimation triggers extreme uncertainty inflation on boundary nodes, pinning error caps against safety ceilings.

This replication pipeline introduces an alternative **Topology-Aware Generalized Linear Mixed Model (GP-GLMM)** using `gpboost` in Python. While both models share a matching `bernoulli_logit` link, continuous geographic kernels (2D vs 3D), and regional random slopes, the current architecture implements a crucial data isolation layer that decouples localized variance profiles by routing historical isolates to independent variance tracks. This insulates the network fields, resolving calculation crashes and rescuing stable typological signals that were previously masked by estimation noise.

## Project Structure

```text
├── output/                             # main results and chart exports
│   ├── feature_synthesis/              # combined dataset for each universal
│   ├── global_isolate_comparisons/     # global mapping views (per universal)
│   ├── isolate_comparisons/            # volatility charts for isolated languages
│   ├── model_predictions/              # results under regional random slope models
│   │
│   ├── global_synthesis_scatter.png                 # comparison scatter plot for all 191 universals
│   ├── GPGLMM_results_191_100tree-3d-group.xlsx     # output from GPGLMM
│   ├── linguistics_connectivity_summary.csv         # density (κ) results
│   ├── linguistics_sensitivity_matrix.csv           # results of spatial sensitivity test
│   ├── parametric_summary.csv                       # results of parameter instability check
│   ├── Results_3D_Master_Synthesis.xlsx             # main spreadsheet sorting rules into final groups
│   ├── Supplementary_Table_S1_Global_Synthesis.xlsx # full parameter database (brms+GPGLMM)
│   ├── Table_A_Consensus.xlsx                       # table of universals agreed by all models
│   ├── Table_B_Expansion.xlsx                       # table of universals found by GPGLMM only
│   ├── universals_forest_plot.pdf                   # forest plot chart (PDF format)
│   └── universals_forest_plot.png                   # forest plot chart (PNG graphic)
├── tlu/                                # raw data and tree files from original study
│   ├── [universal_code]/BT_data.txt                 # coded language features from Grambank for universal
│   ├── [universal_code]/pruned_tree.trees.gz        # historical family tree branch files for universal
│   ├── BT_results_summary.txt                       # universal codes and results from original study
│   └── Glottolog_Languages.csv                      # language metadata from Glottolog
├── utils/                              # background scripts and processing pipelines
│   ├── check_datasets.py                            # data integrity validation check
│   ├── diagnostic_master.py                         # script that sorts rules into matching groups
│   ├── gpglmm_engine.py                             # core Python modeling script
│   └── plotting_master.py                           # chart and graphic generation scripts
├── calculate_connectivity.py           # measures background data density (κ)
├── README.md                           # project documentation
├── run_gpglmm.py                       # main script that fits all models
└── requirements.txt                    # pinned package dependencies
```

## Replication Summary Metrics

The reanalysis confirms **all 60 core universals** that passed the final evolutionary co-evolution checks in the original study. By insulating singleton variance profiles, the current framework maps all 191 universals into five resolution classes:

*   **Cross-framework consensus (83 Rules):** Highly robust features confirmed as significant by both `brms` and `GPGLMM`. This group includes 59 stable baseline patterns and 24 rules that clear regional slope tests (encompassing all 60 final co-evolution universals).
*   **Rescued Universals (16 Rules):** Cross-linguistic patterns that were obscured or dropped by the original `brms` final filters due to parameter instability, recovered via explicit isolate tracking.
*   **Isolate-Driven False Positives (6 Rules):** Typological claims supported by the original `brms` model that collapse into non-significance once background singleton noise is insulated, indicating that their original significance was an artifact of unlinked sample noise.
*   **Consensus Non-Significant (86 Rules):** Universals where both the Bayesian and Frequentist pipelines agree there is no meaningful evolutionary signal.

### Cross-Framework Parameters Summary
By cross-referencing parameters directly within isolated geographic zones, our reanalysis shows that the legacy unconstrained R model experiences calculation failures across **29.8% of all features tested**.

Because both models implement identical continuous spatial maps and regional varying slopes, this volatility is driven entirely by isolate handling. When the legacy model leaves isolated languages unlinked, their parameter explosions pollute the global sampler space. By contrast, anchoring and insulating isolate variance via independent tracks stabilizes the execution, revealing which typological claims are truly robust and which were artifactual:

| Parameter Volatility Failure State | Legacy Bayesian (`brms`) | Topology-Aware (`GP-GLMM`) | Methodological Significance |
| :--- | :---: | :---: | :--- |
| **Stable Execution Profile** | 134 | **191** | Model estimates and intervals ran within normal bounds. |
| **Runaway Uncertainty Explosion** | 27 | 0 | Error margins inflated and pinned against safety ceilings due to missing branch anchors. |
| **Estimate Intercept Collapse** | 8 | 0 | Estimates collapsed entirely, over-smoothing to a global baseline intercept mean. |
| **Concurrent Double Failure** | 22 | 0 | Severe breakdown showing simultaneous estimate flatlining and exploded errors. |
| **Total Features Evaluated** | **191** | **191** |  |

*Data compiled automatically by `utils/diagnostic_master.py` and saved to `output/parametric_summary.csv`.*

## Execution & Replication Pipeline

### Pinned Dependencies
Install the required packages using the requirements file. This automatically downloads the helper package `pykdensity` to manage data density checks:
```bash
pip install -r requirements.txt
```

### Step 1: Audit Dataset Connectivity
Run the density script to measure structural clustering across language family lines and map positions before running models:
```bash
python calculate_connectivity.py
```

### Step 2: Fit the Spatial Models
Run the core script to process raw files, initialize geocentric coordinates, map regional slope arrays, and execute the 191 models:
```bash
python run_gpglmm.py
```

### Step 3: Compile Framework Comparisons
Run the master utility to combine your new Python metrics with the legacy data and sort rules into final groups:
```bash
python utils/diagnostic_master.py
```

### Step 4: Recreate Project Visuals
Run the master plotting utility to output the final scatter comparison plots and domain-specific effect forest plots:
```bash
python utils/plotting_master.py
```
