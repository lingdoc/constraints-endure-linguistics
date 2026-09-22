# Constraints That Endure: How Galton's Problem Exposes Model Bias in Linguistic Typology

This repository contains the code and data tables used to reanalyze the cross-linguistic universals framework from [Verkerk et al. (2025)](https://nature.com).

The current architecture implements a 3D spatial Gaussian Process (GP-GLMM). This pipeline fixes boundary errors for historically isolated languages, uncovering hidden universal constraints caused by parameter collapse/explosion in the original Bayesian model.

---

## Repository Overview

```text
├── output/
│   ├── feature_synthesis/                  # Combined results for each linguistic feature
│   ├── global_isolate_comparisons/         # Broad map visual of all languages
│   ├── isolate_comparisons/                # Detailed comparison plots for isolated languages
│   ├── model_predictions_2d/               # 2D map model summaries (Parquet format)
│   ├── model_predictions_3d/               # 3D map model summaries (Parquet format)
│   │
│   ├── global_synthesis_scatter.png        # Comparison scatter plot for all universals
│   ├── GPGLMM_results_191_100tree-2d.xlsx  # Summary metrics for 2D map model
│   ├── GPGLMM_results_191_100tree-3d.xlsx  # Summary metrics for 3D map model
│   ├── Results_3D_Master_Synthesis.xlsx    # Main data sheet sorting features into 5 groups
│   ├── Supplementary_Table_S1_Global..xlsx # Full data appendix
│   ├── Table_A_Consensus.xlsx              # Universals confirmed by all models
│   ├── Table_B_Expansion.xlsx              # New universals found only by this model
│   ├── universals_forest_plot.pdf          # Forest plot pdf
│   └── universals_forest_plot.png          # Forest plot graphic
│
├── tlu/                                    # Raw data (in subfolders) from the original study
│   ├── BT_results_summary.txt              # Summary datasheet from the original study (strings `bmrs` > `brms`)
│   └── Glottolog_Languages.csv             # Summary glottolog language sheet from the original study
│
├── utils/
│   ├── check_datasets.py                   # Data integrity check before running models
│   ├── diagnostic_master.py                # Main script sorting data into matching groups
│   ├── gpglmm_engine.py                    # Core modeling script
│   └── plotting_master.py                  # Generates all charts and forest plots
│
├── .gitignore
├── README.md
└── run_gpglmm.py                           # Main script to run the entire pipeline
```

---

## How to Run the Pipeline

Running the master script from the project root pulls raw data from the `tlu/` folder (cloned from the original study at https://github.com/SimonGreenhill/TestingLinguisticUniversals) and builds the new models:

```bash
python run_gpglmm.py
```

### Regenerating the Charts and Tables
To update the summary spreadsheets or refresh the charts, run these two utility scripts from the project root in this exact order:

```bash
# 1. Match up the model outputs and sort features into their final groups
python utils/diagnostic_master.py

# 2. Recreate the comparison scatter plot and the main forest plots
python utils/plotting_master.py
```

### Script explanations:
1. **Validation (`utils/check_datasets.py`):** Checks the raw data files in `tlu/` to make sure there are no formatting or structural errors before modeling.
2. **Modeling Engine (`utils/gpglmm_engine.py`):** Runs the main 3D spatial models. *Note: Large output files are saved as compact binary Parquet files in `output/model_predictions_3d/` to save space. Combined results from both the current model and the original study's brms outputs are saved in `output/feature_synthesis/` as CSV files.*
3. **Data Sorting (`utils/diagnostic_master.py`):** Sorts all 191 linguistic features into 5 distinct groups based on how well the models agree, prints the final tally to the terminal, and exports individual sub-tables for the paper.
4. **Plotting (`utils/plotting_master.py`):** Uses the sorted data sheets to draw the paper's final charts, including the main forest plots and the global comparison scatter plot.

---

## Key Findings Preview

### Cross-framework comparison
This scatter plot maps out where the GP-GLMM agrees or disagrees with previous methods across all 191 linguistic features. Features are spaced out horizontally by how much the current model reduced background uncertainty, and grouped into distinct vertical rows so the different categories don't overlap and blur together:

![Meta-Analysis Comparison Map](./output/global_synthesis_scatter.png)

*   **Stable Core Consensus (Deep Blue Circles):** 60 features confidently confirmed by both old and new modeling approaches.
*   **Rescued Universals (brms + 3D) (Royal Blue Triangles):** 28 intermediate features saved after the original study's final checks dropped them.
*   **Rescued Universals (3D Alone) (Cyan Triangles):** 25 brand-new universal rules discovered exclusively by the GP-GLMM.
*   **Projection Shift Artifacts (Crimson Red Crosses):** Features that looked like true universals on flat maps but were proven to be geographical side effects by GP-GLMM.
*   **Consensus Non-Significant (Muted Yellow Squares):** Background traits where all models agree there is no meaningful pattern.

### Distribution of Validated Universals
The forest plot displays estimated model effects (\(\beta\) coefficients) and 95% confidence intervals for all 113 confirmed universals, split by language domain and color-coded to match final groups:

![Forest Plot](./output/universals_forest_plot.png)
