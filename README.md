# Constraints That Endure: How Galton's Problem Exposes Model Bias in Linguistic Typology

This repository provides code and matrices for a parametric reanalysis of the cross-linguistic universals framework described in [Verkerk et al (2025)](https://www.nature.com/articles/s41562-025-02325-z).

The pipeline replaces unanchored multi-level tree priors with a spatial Gaussian Process Generalized Linear Mixed Model (GP-GLMM) via Laplace approximation mode estimation, resolving boundary artifacts across genealogical isolates to recover hidden structural constraints.

---

## Repository overview

```text
├── output/
│   ├── feature_synthesis/                  # Feature-by-feature spatiophylogenetic tables
│   ├── global_isolate_comparisons/         # Macro-level visualization of languages
│   ├── isolate_comparisons/                # Isolate diagnostic comparison plots
│   ├── model_predictions_2d/               # Population latent effects (2D projection)
│   ├── model_predictions_3d/               # Population latent effects (3D Cartesian Parquet)
│   │
│   ├── global_synthesis_scatter.png        # Global framework diagnostic scatter plot
│   ├── GPGLMM_results_191_100tree-2d.xlsx  # Core 2D metric flat projection summaries
│   ├── GPGLMM_results_191_100tree-3d.xlsx  # Core 3D metric chord summaries
│   ├── Results_3D_Master_Synthesis.xlsx    # Master 5-group resolution taxonomy matrix
│   ├── Supplementary_Table_S1_Global..xlsx # Data appendix
│   ├── Table_A_Consensus.xlsx              # Shared cross-model traits
│   ├── Table_B_Expansion.xlsx              # Rescued universals (GP-GLMM only)
│   ├── universals_forest_plot.pdf          # Forest plot
│   └── universals_forest_plot.png          # Forest graphic
│
├── tlu/                                    # Baseline dataset tables
├── utils/
│   ├── check_datasets.py                   # Pre-flight data integrity validation
│   ├── diagnostic_master.py                # Grouping and sub-table synthesizer
│   ├── gpglmm_engine.py                    # Core spatial field optimization engine
│   └── plotting_master.py                  # Graphics & forest plot generator
│
├── .gitignore
├── README.md
└── run_gpglmm.py                           # Pipeline orchestrator
```

---

## Workflow

The repository is built as a linear pipeline. Running the master script from the project root builds models using data from the `tlu` directory, which contains files from the original study (https://github.com/SimonGreenhill/TestingLinguisticUniversals):

```bash
python run_gpglmm.py
```

To check the outputs or generate plots, run the scripts in the `utils` directory.

### Notes on scripts:
1. **Validation (`utils/check_datasets.py`):** Audits the structural integrity of baseline typological files in the `tlu/` directory.
2. **Model Process (`utils/gpglmm_engine.py`):** Instantiates the parametric spatial kernels across coordinate metrics. *(Note: Full population spatial predictions are stored in binary Snappy Parquet blocks inside `output/model_predictions_3d/` to optimize repository storage). Synthesized results combined with the `brms` outputs from the original study are found at `output/feature_synthesis/`*
3. **Summarization (`utils/diagnostic_master.py`):** Computes the 5-group taxonomy, prints the cross-model alignment assertions to the terminal, and exports sub-tables (`Table_A_Consensus.xlsx`, `Table_B_Expansion.xlsx`).
4. **Plots (`utils/plotting_master.py`):** Evaluates compiled parameter variances to output the global isolate continuum charts and the primary domain-segregated forest plot graphics.

---

## Key Findings Preview

### Proportional Grid Distribution of Validated Universals
The forest plot displays estimated fixed-effect slope parameters ($\beta$ coefficients) and 95% confidence intervals for the 113 spatial discoveries, stratified by typological domain and color-coded by cross-framework resolution class:

![Forest Plot](./output/universals_forest_plot.png)
