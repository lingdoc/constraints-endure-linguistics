"""
calculate_connectivity.py

This script checks data clustering across the linguistic dataset layers.
It measures spatial distances and historical tree structures across feature folders.
"""

import os
import glob
import pandas as pd
import numpy as np
from pytstability import calculate_densities

# Set up project directories and output path
data_dir = "tlu"
output_csv = os.path.join("output", "linguistics_connectivity_summary.csv")
os.makedirs("output", exist_ok=True)

print("\nStarting data connectivity analysis...")

glottolog_path = os.path.join(data_dir, "Glottolog_Languages.csv")
compiled_results = []

total_valid_spatial_features = 0
total_valid_structural_features = 0
total_valid_tsi_features = 0
running_weighted_spatial_sum = 0.0
running_weighted_structural_sum = 0.0
running_weighted_tsi_sum = 0.0

if not os.path.exists(glottolog_path):
    print(f"Error: Master map file missing at {glottolog_path}")
    exit(1)

print(f"Loading global language coordinates from: {glottolog_path}")
gldf = pd.read_csv(glottolog_path)

gldf = gldf.dropna(subset=['latitude', 'longitude'])
gldf_shared = gldf[['glottocode', 'macroarea', 'Family_ID', 'longitude', 'latitude']].copy()
gldf_shared['glottocode'] = gldf_shared['glottocode'].astype(str).str.strip().str.lower()

feat_data_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*data.txt")))

for feat_path in feat_data_paths:
    feat_id = str(os.path.basename(os.path.dirname(feat_path)))

    feat_dir = os.path.dirname(feat_path)
    tree_matches = glob.glob(os.path.join(feat_dir, "*trees.gz"))
    tree_path = tree_matches[0] if tree_matches else None

    print(f"\nProcessing feature layer: {feat_id}")

    fdf = pd.read_csv(feat_path, delimiter="\t", header=None, names=["glottocode", "DV", "IV"])

    if len(fdf) < 10:
        print(f"    Skipping: Not enough data rows (found {len(fdf)}, needs at least 10)")
        continue

    fdf['glottocode'] = fdf['glottocode'].astype(str).str.strip().str.lower()
    ling_df = pd.merge(gldf_shared, fdf, on='glottocode', how='inner')
    sample_size = len(ling_df)

    print(f"    Matched {sample_size} languages on the map.")
    print(f"    Family tree file: {tree_path}")

    try:
        t_metrics = calculate_densities(
            data=ling_df,
            id_col='glottocode',
            tree=tree_path,
            tree_type='adaptive',
            coord_cols=['latitude', 'longitude'],
            spatial_cats='macroarea' if 'macroarea' in ling_df.columns else None,
            struct_cats='Family_ID' if 'Family_ID' in ling_df.columns else None,
            spatial_km=500.0,
            struct_depth=8,
            verbose=False
        )

        k_spatial = float(t_metrics["spatial_metrics"].get("kappa", np.nan))
        k_structural = float(t_metrics["structural_metrics"].get("kappa", np.nan))
        tsi_current = float(t_metrics.get("topology_stability_index", np.nan))
        gamma_current = float(t_metrics.get("critical_horizon", np.nan))

    except Exception as e:
        print(f"    Warning: Density calculation failed for {feat_id}: {e}")
        k_spatial = np.nan
        k_structural = np.nan
        tsi_current = np.nan
        gamma_current = np.nan

    if not pd.isna(tsi_current) and tsi_current <= 0.0018:
        tier_status = "critical funnel zone"
    elif not pd.isna(tsi_current) and tsi_current <= 0.0500:
        tier_status = "moderate geometric shear"
    else:
        tier_status = "stable pooling backbone"

    compiled_results.append({
        "Feature": feat_id,
        "Sample_Size_N": sample_size,
        "Spatial_Density": k_spatial if not pd.isna(k_spatial) else np.nan,
        "Structural_Density": k_structural if not pd.isna(k_structural) else np.nan,
        "Topology_Stability_Index": tsi_current if not pd.isna(tsi_current) else np.nan,
        "Critical_Horizon": gamma_current if not pd.isna(gamma_current) else np.nan,
        "Operational_Status": tier_status
    })

    if not pd.isna(k_spatial):
        total_valid_spatial_features += sample_size
        running_weighted_spatial_sum += (k_spatial * sample_size)

    if not pd.isna(k_structural):
        total_valid_structural_features += sample_size
        running_weighted_structural_sum += (k_structural * sample_size)

    if not pd.isna(tsi_current):
        total_valid_tsi_features += sample_size
        running_weighted_tsi_sum += tsi_current

print("\nConnectivity check complete")
print("-" * 80)

if compiled_results:
    # Target directory verification check
    df_out = pd.DataFrame(compiled_results)
    df_out.to_csv(output_csv, index=False)
    print(f"Summary file saved to: {output_csv}\n")

    print(df_out.to_string(index=False, formatters={
        "Spatial_Density": lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A",
        "Structural_Density": lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A",
        "Topology_Stability_Index": lambda x: f"{x:.5f}" if not pd.isna(x) else "N/A"
    }))
    print("-" * 80)

    if total_valid_spatial_features > 0:
        avg_spatial = running_weighted_spatial_sum / total_valid_spatial_features
        print(f"Overall dataset spatial density score    : {avg_spatial:.4f}")
    else:
        print("Overall dataset spatial density score    : N/A")

    if total_valid_structural_features > 0:
        avg_structural = running_weighted_structural_sum / total_valid_structural_features
        print(f"Overall dataset structural density score : {avg_structural:.4f}")
    else:
        print("Overall dataset structural density score : N/A")

    if total_valid_tsi_features > 0:
        avg_tsi = running_weighted_tsi_sum / total_valid_tsi_features
        print(f"Overall dataset structural density score : {avg_tsi:.4f}")
    else:
        print("Overall dataset structural density score : N/A")
else:
    print("\nError: No valid features were compiled.")
print("-" * 80 + "\n")
