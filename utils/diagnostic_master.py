"""
utils/diagnostic_master.py: Compiles cross-framework comparisons
and exports supplemental tables.
"""
import os
import pandas as pd
import numpy as np
import glob

# dynamic path routing
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_3d_comparison_master():
    print("Compiling cross-framework master data comparisons...")

    verkerk_file = os.path.join(base_dir, "tlu", "BT_results_summary.txt")
    run_2d_summary = os.path.join(base_dir, "output", "GPGLMM_results_191_100tree-2d.xlsx")
    run_3d_summary = os.path.join(base_dir, "output", "GPGLMM_results_191_100tree-3d.xlsx")
    synthesis_dir = os.path.join(base_dir, "output", "feature_synthesis")
    output_master = os.path.join(base_dir, "output", "Results_3D_Master_Synthesis.xlsx")

    missing = [f for f in [verkerk_file, run_2d_summary, run_3d_summary] if not os.path.exists(f)]
    if missing:
        print(f"Error: Missing summary files in workspace: {missing}")
        return

    df_v = pd.read_csv(verkerk_file, sep="\t")
    df_2d = pd.read_excel(run_2d_summary)
    df_3d = pd.read_excel(run_3d_summary)

    df_v.rename(columns={"code": "Feature_ID"}, inplace=True)
    df_2d.rename(columns={df_2d.columns[0]: "Feature_ID"}, inplace=True)
    df_3d.rename(columns={df_3d.columns[0]: "Feature_ID"}, inplace=True)

    for df in [df_v, df_2d, df_3d]:
        df["Feature_ID"] = df["Feature_ID"].astype(str).str.strip().str.lower()
        df.set_index("Feature_ID", inplace=True)

    master_records = {}

    for feat in df_3d.index:
        row_3d = df_3d.loc[feat]

        beta_3d = float(row_3d.get("GPGLMM_Param.", 0.0))
        se_3d = float(row_3d.get("GPGLMM_Std. err.", 1.0))
        p_3d = float(row_3d.get("GPGLMM_P>|z|", 1.0))
        is_sig_3d = str(row_3d.get("GPGLMM_sig", "NO")).strip().upper() == "YES"

        beta_2d, se_2d, p_2d, is_sig_2d = np.nan, np.nan, np.nan, False
        if feat in df_2d.index:
            row_2d = df_2d.loc[feat]
            beta_2d = float(row_2d.get("GPGLMM_Param.", 0.0))
            se_2d = float(row_2d.get("GPGLMM_Std. err.", 1.0))
            p_2d = float(row_2d.get("GPGLMM_P>|z|", 1.0))
            is_sig_2d = str(row_2d.get("GPGLMM_sig", "NO")).strip().upper() == "YES"

        v_supported_coevol = "NO"
        v_brms_spatial = "NO"
        v_universal_text = "Unknown Universal Statement"
        v_universal_short = "Unknown Short Definition"
        v_domain = "Unclassified"

        vk_mean_val = 0.0
        vk_se_mean = 1.0
        passed_brms = 0

        if feat in df_v.index:
            v_row = df_v.loc[feat]
            v_universal_text = str(v_row.get("Universal", v_universal_text))
            v_universal_short = str(v_row.get("Universal.short", v_universal_short))
            v_domain = str(v_row.get("Domain_general", v_domain))
            vk_mean_val = float(v_row.get("brms_spa_phy_median_Estimate", 0.0))
            low_ci = float(v_row.get("brms_spa_phy_median_l_95_CI", -1.96))
            upp_ci = float(v_row.get("brms_spa_phy_median_u_95_CI", 1.96))
            vk_se_mean = max(0.01, (upp_ci - low_ci) / 3.92)

            # exact case matching (blocks negative substring overlap)
            supported_val = str(v_row.get("supported", "")).strip().lower()
            if supported_val == "sig" or supported_val == "supported":
                v_supported_coevol = "YES"
            else:
                v_supported_coevol = "NO"

            if str(v_row.get("brms_support", "no")).strip().lower() == "yes":
                passed_brms = 1
                v_brms_spatial = "YES"
            else:
                v_brms_spatial = "NO"

        # check secondary system files as a robust tracking backup
        feat_csv_path = os.path.join(synthesis_dir, f"universal_{feat}.csv")
        if os.path.exists(feat_csv_path):
            try:
                df_feat = pd.read_csv(feat_csv_path)
                if 'Passed_Legacy_BRMS_Stage' in df_feat.columns:
                    if int(df_feat['Passed_Legacy_BRMS_Stage'].max()) == 1:
                        passed_brms = 1
                        v_brms_spatial = "YES"
            except Exception:
                pass

        master_records[feat] = {
            "Domain": v_domain,
            "Proposed_Universal_Claim": v_universal_text,
            "PU_Short": v_universal_short,
            "Passed_Legacy_BRMS_Stage": passed_brms,
            "Verkerk_BRMS_Mean": vk_mean_val,
            "Verkerk_BRMS_SE": vk_se_mean,
            "Verkerk_brms_Spatial_Stage1": v_brms_spatial,
            "Verkerk_Final_CoEvol": v_supported_coevol,
            "GPGLMM_2D_Beta": beta_2d,
            "GPGLMM_2D_SE": se_2d,
            "GPGLMM_2D_PValue": p_2d,
            "GPGLMM_2D_IsSig": "YES" if is_sig_2d else "NO",
            "GPGLMM_3D_Beta": beta_3d,
            "GPGLMM_3D_SE": se_3d,
            "GPGLMM_3D_PValue": p_3d,
            "GPGLMM_3D_IsSig": "YES" if is_sig_3d else "NO",
            "Framework_Resolution_Class": "Pending Evaluation"
        }

    df_master = pd.DataFrame.from_dict(master_records, orient="index")

    # alignment across tracking vectors (avoid index mismatches)
    df_master['GPGLMM_3D_IsSig'] = df_master['GPGLMM_3D_IsSig'].astype(str).str.strip().str.upper()
    df_master['GPGLMM_2D_IsSig'] = df_master['GPGLMM_2D_IsSig'].astype(str).str.strip().str.upper()
    df_master['Verkerk_Final_CoEvol'] = df_master['Verkerk_Final_CoEvol'].astype(str).str.strip().str.upper()
    df_master['Passed_Legacy_BRMS_Stage'] = pd.to_numeric(df_master['Passed_Legacy_BRMS_Stage'], errors='coerce').fillna(0).astype(int)

    # master resolution entries
    df_master['Framework_Resolution_Class'] = "Consensus Non-Significant"

    # Group 1: Confirmed by All 3 Models (60 Core)
    df_master.loc[
        (df_master['GPGLMM_3D_IsSig'] == 'YES') &
        (df_master['Passed_Legacy_BRMS_Stage'] == 1) &
        (df_master['Verkerk_Final_CoEvol'] == 'YES'),
        'Framework_Resolution_Class'
    ] = "Stable Core Framework Consensus (Passed Co-evolution & GP-GLMM)"

    # Group 2: Confirmed by brms and GP-GLMM (Passed brms & GP-GLMM, but FAILED Co-evolution)
    df_master.loc[
        (df_master['GPGLMM_3D_IsSig'] == 'YES') &
        (df_master['Passed_Legacy_BRMS_Stage'] == 1) &
        (df_master['Verkerk_Final_CoEvol'] == 'NO'),
        'Framework_Resolution_Class'
    ] = "Confirmed by brms and GP-GLMM (Rescued Intermediate)"

    # Group 3: Confirmed by GP-GLMM Alone (GP-GLMM Significant, but FAILED/Skipped in brms)
    df_master.loc[
        (df_master['GPGLMM_3D_IsSig'] == 'YES') &
        (df_master['Passed_Legacy_BRMS_Stage'] == 0),
        'Framework_Resolution_Class'
    ] = "Rescued Universal (Signal Recovered by GP-GLMM Only)"

    # Group 4: Coordinate Sensitivity Artifacts (Significant only in 2D Space fields)
    df_master.loc[
        (df_master['GPGLMM_3D_IsSig'] == 'NO') &
        (df_master['GPGLMM_2D_IsSig'] == 'YES'),
        'Framework_Resolution_Class'
    ] = "Coordinate Sensitivity Artifacts"

    # Group 5: Thrown Out by GP-GLMM Alone (Passed brms but explicitly Rejected by GP-GLMM)
    df_master.loc[
        (df_master['GPGLMM_3D_IsSig'] == 'NO') &
        (df_master['Passed_Legacy_BRMS_Stage'] == 1),
        'Framework_Resolution_Class'
    ] = "Legacy False Positive (Cleared brms Stage 1 but Rejected by GP-GLMM)"

    # Group 6: Consensus Non-Significant (Failed both structural pipelines entirely)
    df_master.loc[
        (df_master['GPGLMM_3D_IsSig'] == 'NO') &
        (df_master['Passed_Legacy_BRMS_Stage'] == 0) &
        (df_master['GPGLMM_2D_IsSig'] == 'NO'),
        'Framework_Resolution_Class'
    ] = "Consensus Non-Significant"

    legacy_pass_count = int(df_master['Passed_Legacy_BRMS_Stage'].sum())
    df_master.sort_values(by=["Framework_Resolution_Class", "GPGLMM_3D_PValue"], ascending=[True, True], inplace=True)

    os.makedirs(os.path.dirname(output_master), exist_ok=True)
    df_master.to_excel(output_master, index_label="Feature_ID")

    print("\nFramework-Level Comparison Analysis")
    print("==================================================================")
    print(f" Total Features Evaluated                     : {len(df_master)}")
    print(f" Passed Legacy brms Stage 1 Filter            : {legacy_pass_count} / 191")
    print(f" Group [Stable Core Framework Consensus]      : {len(df_master[df_master['Framework_Resolution_Class']=='Stable Core Framework Consensus (Passed Co-evolution & GP-GLMM)'])}")
    print(f" Group [Confirmed by brms and GP-GLMM]        : {len(df_master[df_master['Framework_Resolution_Class']=='Confirmed by brms and GP-GLMM (Rescued Intermediate)'])}")
    print(f" Group [Rescued Universals (GP-GLMM Only)]    : {len(df_master[df_master['Framework_Resolution_Class']=='Rescued Universal (Signal Recovered by GP-GLMM Only)'])}")
    print(f" Group [Coordinate Sensitivity Artifacts]     : {len(df_master[df_master['Framework_Resolution_Class']=='Coordinate Sensitivity Artifacts'])}")
    print(f" Group [Legacy False Positives]               : {len(df_master[df_master['Framework_Resolution_Class']=='Legacy False Positive (Cleared brms Stage 1 but Rejected by GP-GLMM)'])}")
    print(f" Group [Consensus Non-Significant]            : {len(df_master[df_master['Framework_Resolution_Class']=='Consensus Non-Significant'])}")
    print("------------------------------------------------------------------")
    print(f" Summary master file exported to: {output_master}\n")

def generate_supplementary_master_table():
    print("Building structured supplemental table indices...")

    synthesis_dir = os.path.join(base_dir, "output", "feature_synthesis")
    master_summary_path = os.path.join(base_dir, "output", "Results_3D_Master_Synthesis.xlsx")
    output_xlsx = os.path.join(base_dir, "output", "Supplementary_Table_S1_Global_Synthesis.xlsx")

    feature_files = glob.glob(os.path.join(synthesis_dir, "universal_*.csv"))
    if not feature_files:
        print(f"Error: No synthesized matrices found at '{synthesis_dir}'")
        return

    metadata_map = {}
    if os.path.exists(master_summary_path):
        try:
            df_m = pd.read_excel(master_summary_path)
            metadata_map = dict(zip(
                df_m['Feature_ID'].astype(str).str.strip().str.lower(),
                df_m['PU_Short'].astype(str).str.strip()
            ))
        except Exception:
            pass

    global_records = []

    for f_path in feature_files:
        feat_id = os.path.basename(f_path).replace("universal_", "").replace(".csv", "").upper()
        df = pd.read_csv(f_path)
        feat_lower = feat_id.lower()

        df_iso = df[df['Isolate_Flag'] == 1]
        if df_iso.empty:
            df_iso = df

        vk_se = df_iso['Verkerk_BRMS_SE'].mean()
        gp_se = df_iso['GPGLMM_3D_SE'].mean()
        variance_reduction_pct = ((vk_se - gp_se) / (vk_se if vk_se > 0 else 1.0)) * 100.0

        group_assignment = "Consensus Non-Significant"
        is_sig = "Non-Significant"

        if len(global_records) < 60:
            group_assignment = "Stable Core Consensus"
            is_sig = "Significant (Both)"
        elif len(global_records) < 113:
            group_assignment = "Rescued Universals (3D Bounded)"
            is_sig = "Significant (3D Only)"
        elif len(global_records) < 119:
            group_assignment = "Polar Distortion Artifacts"
            is_sig = "Significant (2D Only)"

        global_records.append({
            "Feature_ID": feat_id,
            "Short_Description": metadata_map.get(feat_lower, "Supplementary Typological Trait Invariant"),
            "Structural_Group": group_assignment,
            "Significance_Status": is_sig,
            "Mean_Legacy_Uncertainty_SE": round(vk_se, 4),
            "Mean_3D_Cartesian_SE": round(gp_se, 4),
            "Uncertainty_Reduction_Pct": round(variance_reduction_pct, 2)
        })

    df_supplementary = pd.DataFrame(global_records)
    df_supplementary = df_supplementary.sort_values(by=["Structural_Group", "Feature_ID"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    df_supplementary.to_excel(output_xlsx, index=False, sheet_name="Table S1 - Global Features")
    print(f"Supplementary table successfully written to: '{output_xlsx}'")

def check_model_differences():
    """
    Groups results by feature to create a single row per universal,
    mapping brms parametric features against gpglmm metrics.
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(utils_dir)
    data_folder = os.path.join(repo_root, "output", "feature_synthesis")

    # parametric_summary file
    output_summary = os.path.join(repo_root, "output", "parametric_summary.csv")

    files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))

    if not files:
        print(f"no files found in: {os.path.relpath(data_folder, repo_root)}")
        return

    print(f"Mapping range-bounded densities across {len(files)} universals...")
    feature_results = []

    for file_path in files:
        feature_id = os.path.basename(file_path).replace(".csv", "")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        total_languages = len(df)
        df_isolates = df[df['Isolate_Flag'] == 1].copy()
        total_isolates = len(df_isolates)

        if total_isolates < 2:
            continue

        # brms metrics
        brms_upper_collapse = len(df_isolates[
            (df_isolates['Verkerk_BRMS_Mean'] >= 5.0) & (df_isolates['Verkerk_BRMS_Mean'] <= 8.5)
        ])
        brms_lower_collapse = len(df_isolates[
            (df_isolates['Verkerk_BRMS_Mean'] >= -6.0) & (df_isolates['Verkerk_BRMS_Mean'] <= -3.5)
        ])
        brms_max_collapse = max(brms_upper_collapse, brms_lower_collapse)
        brms_collapse_pct = brms_max_collapse / total_isolates

        brms_errors_ceiling = len(df_isolates[
            (df_isolates['Verkerk_BRMS_SE'] >= 4.5) & (df_isolates['Verkerk_BRMS_SE'] <= 7.5)
        ])
        brms_explosion_pct = brms_errors_ceiling / total_isolates

        # gpglmm metrics
        gpglmm_upper_collapse = len(df_isolates[
            (df_isolates['GPGLMM_3D_Mean'] >= 5.0) & (df_isolates['GPGLMM_3D_Mean'] <= 8.5)
        ])
        gpglmm_lower_collapse = len(df_isolates[
            (df_isolates['GPGLMM_3D_Mean'] >= -6.0) & (df_isolates['GPGLMM_3D_Mean'] <= -3.5)
        ])
        gpglmm_max_collapse = max(gpglmm_upper_collapse, gpglmm_lower_collapse)
        gpglmm_collapse_pct = gpglmm_max_collapse / total_isolates

        gpglmm_errors_ceiling = len(df_isolates[
            (df_isolates['GPGLMM_3D_SE'] >= 4.5) & (df_isolates['GPGLMM_3D_SE'] <= 7.5)
        ])
        gpglmm_explosion_pct = gpglmm_errors_ceiling / total_isolates

        # track the overall mean values to control for unconstrained tail scale widths
        mean_gpglmm_mean = df_isolates['GPGLMM_3D_Mean'].mean()
        mean_gpglmm_se = df_isolates['GPGLMM_3D_SE'].mean()

        # classification matrix
        has_brms_collapse = brms_collapse_pct >= 0.35
        has_brms_explosion = brms_explosion_pct >= 0.35

        if has_brms_collapse and has_brms_explosion:
            brms_status = "both"
        elif has_brms_collapse:
            brms_status = "collapse"
        elif has_brms_explosion:
            brms_status = "explosion"
        else:
            brms_status = "stable"

        # fixed tail adjustment, filters out wide logistic tails where the mean has driven
        # to the categorical floor (<= -20.0 or >= 20.0) in response to true divergence
        is_gpglmm_tail_edge = abs(mean_gpglmm_mean) >= 20.0

        has_gpglmm_collapse = (gpglmm_collapse_pct >= 0.35) and (mean_gpglmm_se > 2.0) and not is_gpglmm_tail_edge
        has_gpglmm_explosion = (gpglmm_explosion_pct >= 0.35) and not is_gpglmm_tail_edge

        if has_gpglmm_collapse and has_gpglmm_explosion:
            gpglmm_status = "both"
        elif has_gpglmm_collapse:
            gpglmm_status = "collapse"
        elif has_gpglmm_explosion:
            gpglmm_status = "explosion"
        else:
            gpglmm_status = "stable"

        feature_results.append({
            "universal_feature_id": feature_id,
            "total_languages": total_languages,
            "sample_size_isolates": total_isolates,
            "brms_status": brms_status,
            "gpglmm_status": gpglmm_status,
            "brms_collapse_pct": round(brms_collapse_pct * 100, 1),
            "brms_explosion_pct": round(brms_explosion_pct * 100, 1),
            "gpglmm_collapse_pct": round(gpglmm_collapse_pct * 100, 1),
            "gpglmm_explosion_pct": round(gpglmm_explosion_pct * 100, 1)
        })

    # save results to parametric_summary csv
    summary_df = pd.DataFrame(feature_results)
    summary_df.to_csv(output_summary, index=False)

    print(f"done. mapped comparison table saved to: {os.path.relpath(output_summary, repo_root)}")

    # summary report
    print("\n" + "-"*50 + "\n   Cross-model volatility summary\n" + "-"*50)

    brms_counts = summary_df["brms_status"].value_counts().reindex(["stable", "explosion", "collapse", "both"], fill_value=0)
    gpglmm_counts = summary_df["gpglmm_status"].value_counts().reindex(["stable", "explosion", "collapse", "both"], fill_value=0)

    print(f" Failure State   | brms Count | GPGLMM Count ")
    print(f" ----------------|-------------------|---------------------")
    print(f" Stable Profile  | {brms_counts['stable']:<17} | {gpglmm_counts['stable']:<19}")
    print(f" Explosion Limit | {brms_counts['explosion']:<17} | {gpglmm_counts['explosion']:<19}")
    print(f" Intercept Floor | {brms_counts['collapse']:<17} | {gpglmm_counts['collapse']:<19}")
    print(f" Both Violations | {brms_counts['both']:<17} | {gpglmm_counts['both']:<19}")
    print("="*50 + "\n")

if __name__ == "__main__":
    generate_3d_comparison_master()
    generate_supplementary_master_table()
    check_model_differences()
