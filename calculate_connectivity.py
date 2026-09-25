"""
calculate_connectivity.py

this script checks data clustering across the linguistic dataset layers.
it uses the installed 'pykdensity' library to measure spatial distances
and historical family tree branches across all individual feature folders,
then calculates an average score for the whole dataset.
"""

import os
import glob
import pandas as pd
import numpy as np
from pykdensity import calculate_densities

# project setup
# folders to read/write data
data_dir = "tlu"
output_csv = os.path.join("output", "linguistics_connectivity_summary.csv")
os.makedirs("output", exist_ok=True)

print("\nStarting data connectivity analysis")

glottolog_path = os.path.join(data_dir, "Glottolog_Languages.csv")
compiled_results = []

total_valid_spatial_features = 0
total_valid_structural_features = 0
running_weighted_spatial_sum = 0.0
running_weighted_structural_sum = 0.0

# 1. load coordinate master tracking data
if not os.path.exists(glottolog_path):
    print(f"Error: Master map file missing at {glottolog_path}")
    exit(1)

print(f"Loading global language coordinates from: {glottolog_path}")
gldf = pd.read_csv(glottolog_path)

# drop rows missing latitude or longitude to ensure stable distance checks
gldf = gldf.dropna(subset=['latitude', 'longitude'])
gldf_shared = gldf[['glottocode', 'macroarea', 'Family_ID', 'longitude', 'latitude']].copy()
gldf_shared['glottocode'] = gldf_shared['glottocode'].astype(str).str.strip().str.lower()

# nested feature folders processing loop
# locate the individual feature text files sitting inside their separate subfolders
feat_data_paths = sorted(glob.glob(os.path.join(data_dir, "*", "*data.txt")))

for feat_path in feat_data_paths:
    # extract the parent subfolder name to label the feature (e.g., '0008KA')
    feat_id = str(os.path.basename(os.path.dirname(feat_path)))

    # search the same subfolder for the paired compressed family tree file
    feat_dir = os.path.dirname(feat_path)
    tree_matches = glob.glob(os.path.join(feat_dir, "*trees.gz"))
    tree_path = tree_matches[0] if tree_matches else None

    print(f"\nProcessing feature layer: {feat_id}")

    # load the feature text file
    fdf = pd.read_csv(feat_path, delimiter="\t", header=None, names=["glottocode", "DV", "IV"])

    # skip features that do not have enough data points to be meaningful
    if len(fdf) < 10:
        print(f"    Skipping: Not enough data rows (found {len(fdf)}, needs at least 10)")
        continue

    # align the feature rows with our coordinate master file using language codes
    fdf['glottocode'] = fdf['glottocode'].astype(str).str.strip().str.lower()
    ling_df = pd.merge(gldf_shared, fdf, on='glottocode', how='inner')
    sample_size = len(ling_df)

    print(f"    Matched {sample_size} languages on the map.")
    print(f"    Family tree file: {tree_path}")

    # run the calculator function using map coordinates and dynamic tree percentages
    # the package prints individual data layer diagnostics automatically
    k_spatial, k_structural = calculate_densities(
        data=ling_df,
        id_col='glottocode',
        tree=tree_path,
        tree_type='adaptive',  # uses relative percentages to balance uneven branches
        coord_cols=['latitude', 'longitude'],
        spatial_threshold_km=500.0,   # flags points within 500km as connected
        structural_depth_threshold=8,  # limits connections to sub-families
        verbose=True
    )

    # store findings for the final table
    compiled_results.append({
        "Feature": feat_id,
        "Sample_Size_N": sample_size,
        "Spatial_Density": k_spatial if not pd.isna(k_spatial) else "N/A",
        "Structural_Density": k_structural if not pd.isna(k_structural) else "N/A"
    })

    # track counts to ensure small datasets do not skew final averages
    if not pd.isna(k_spatial):
        total_valid_spatial_features += sample_size
        running_weighted_spatial_sum += (k_spatial * sample_size)

    if not pd.isna(k_structural):
        total_valid_structural_features += sample_size
        running_weighted_structural_sum += (k_structural * sample_size)

# save and show summary logs
print("\nConnectivity check complete")
print("-" * 80)

if compiled_results:
    df_out = pd.DataFrame(compiled_results)
    df_out.to_csv(output_csv, index=False)
    print(f"Summary file saved to: {output_csv}\n")

    # print the final table
    print(df_out.to_string(index=False))
    print("-" * 80)

    # calculate global compound dataset averages
    if total_valid_spatial_features > 0:
        avg_spatial = running_weighted_spatial_sum / total_valid_spatial_features
        print(f"Dataset average spatial density score    : {avg_spatial:.4f}")
    else:
        print("Dataset average spatial density score    : N/A")

    if total_valid_structural_features > 0:
        avg_structural = running_weighted_structural_sum / total_valid_structural_features
        print(f"Dataset average structural density score : {avg_structural:.4f}")
    else:
        print("Dataset average structural density score : N/A")
else:
    print("\nError: No valid features were compiled.")
print("-" * 80 + "\n")
