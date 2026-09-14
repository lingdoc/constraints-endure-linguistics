import os
import sys
import glob
import shutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 8ptimize matrix calculation threads for 32 cores, 4 workers
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# force Python to check active working folder first
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# import the engine function
from utils.gpglmm_engine import process_single_feature_gpglmm

def archive_old_run(target_dir="output/model_predictions", completed_features=None):
    """
    Archives old run matrices. If resuming, it preserves files belonging
    to features that are already marked completed in the checkpoint log.
    """
    if completed_features is None:
        completed_features = set()

    if os.path.exists(target_dir):
        old_files = [f for f in os.listdir(target_dir) if f.endswith('.parquet')]
        if old_files:
            backup_dir = "output/model_predictions_backup"

            # Identify files to clear out vs files to keep based on resume state
            files_to_archive = []
            for f in old_files:
                # Extracts feature name from: gpglmm_latent_distribution_[feature].parquet
                # or gpglmm_100tree_trajectory_[feature].parquet
                feat_token = f.replace("gpglmm_latent_distribution_", "").replace("gpglmm_100tree_trajectory_", "").replace(".parquet", "").upper()
                if feat_token not in completed_features:
                    files_to_archive.append(f)

            if files_to_archive:
                os.makedirs(backup_dir, exist_ok=True)
                print(f"📦 Archiving {len(files_to_archive)} unverified/older run matrices to '{backup_dir}'...")
                for f in files_to_archive:
                    src = os.path.join(target_dir, f)
                    dst = os.path.join(backup_dir, f)
                    shutil.move(src, dst)
                print(" Baseline folder cleared of non-checkpointed artifacts.")
    else:
        os.makedirs(target_dir, exist_ok=True)

def save_checkpoint(xdict, output_excel):
    """
    Writes the currently accumulated dictionary state directly to the production Excel sheets.
    """
    if not xdict:
        return
    df_final = pd.DataFrame.from_dict(xdict, orient='index')
    ordered_cols = [
        "Status", "Reason", "Total_Languages_Found", "Distinct_Macroareas",
        "Distinct_Families", "DV_Variance", "DV_Mean", "GPGLMM_n_obs",
        "GPGLMM_Param.", "GPGLMM_Std. err.", "GPGLMM_z value", "GPGLMM_P>|z|",
        "GPGLMM_sig"
    ]
    available_cols = [c for c in ordered_cols if c in df_final.columns]
    df_final = df_final[available_cols]

    # keep the .xlsx extension at the absolute tail of the string so Pandas matches openpyxl
    temp_excel = output_excel.replace(".xlsx", "_temp.xlsx")

    df_final.to_excel(temp_excel, index_label="Feature_ID")
    os.replace(temp_excel, output_excel)

if __name__ == "__main__":
    output_excel = "output/GPGLMM_results_191_100tree-3d.xlsx"
    predictions_folder = "output/model_predictions"
    os.makedirs("output", exist_ok=True)

    xdict = {}
    completed_features = set()

    # checkpoint ingestion
    temp_check = output_excel.replace(".xlsx", "_temp.xlsx")
    if os.path.isfile(temp_check) and not os.path.isfile(output_excel):
        print(f" Recovering from an interrupted write cycle. Repairing '{temp_check}'...")
        os.replace(temp_check, output_excel)
    elif os.path.isfile(temp_check) and os.path.isfile(output_excel):
        os.remove(temp_check)

    # parse previous metrics sheet if resuming a crashed run
    if os.path.isfile(output_excel):
        try:
            print(f" Found existing production master sheet at '{output_excel}'. Reading checkpoint state...")
            df_checkpoint = pd.read_excel(output_excel, index_col="Feature_ID")

            # Map index row tokens back into memory dict format
            for idx, row in df_checkpoint.iterrows():
                feat_key = str(idx).strip().upper()
                xdict[feat_key] = row.to_dict()
                completed_features.add(feat_key)

            print(f" Successfully recovered {len(completed_features)} completed universals from checkpoint. Resuming run...")
        except Exception as read_err:
            print(f" Warning: Could not parse existing output spreadsheet ({str(read_err)}). Starting fresh.")
            xdict = {}
            completed_features = set()

    # clean and isolate workspace directories while preserving verified resume layers
    archive_old_run(predictions_folder, completed_features)

    print(" Loading core Glottolog geographic and language data tables...")
    gldf = pd.read_csv("tlu/Glottolog_Languages.csv")
    gldf['Family_ID'] = gldf['Family_ID'].replace(['NaN', 'nan', 'None', ''], pd.NA)
    gldf['Family_ID'] = gldf['Family_ID'].fillna(gldf['glottocode']).astype(str).str.strip()
    gldf = gldf.dropna(subset=['latitude'])
    gldf_shared = gldf[['glottocode', 'macroarea', 'Family_ID', 'longitude', 'latitude']].copy()

    print(" Scanning 'tlu/' folders for remaining target tracks...")
    search_pattern = os.path.join("tlu", "*", "*data.txt")
    all_raw_files = sorted(glob.glob(search_pattern))

    # strip out every path that matches an already verified ID
    files_to_process = []
    for ffile in all_raw_files:
        # resolves folder token out of path format: tlu/[FEATURE_ID]/[FEATURE_ID]_data.txt
        clean_path = ffile.replace("\\", "/")
        path_parts = clean_path.split("/")
        feat_id = path_parts[-2].upper()

        if feat_id not in completed_features:
            files_to_process.append(ffile)

    if not all_raw_files:
        print(" Error: Could not locate any raw '*data.txt' feature files inside your 'tlu/' folder structure.")
        exit()

    TARGET_NTREES = 100
    MAX_WORKERS = 4  # Balanced worker allocation for 32 threads total (4 x 8 threads)

    print("\n" + "="*75)
    print(" Launching 191-feature analysis pipeline with 3D coordinates")
    print("="*75)
    print(f" Initial Repository Pool : {len(all_raw_files)} Total Universals")
    print(f" Already Completed Runs  : {len(completed_features)} Universals [SKIPPED]")
    print(f" Remaining Active Queue  : {len(files_to_process)} Universals to Process")
    print(f" System Engine Configuration : Spinning up {MAX_WORKERS} processes ({int(32/MAX_WORKERS)} cores each)")
    print(f" Tree Sampling Density   : Evaluating ALL {TARGET_NTREES} tree matrices per feature\n")

    if not files_to_process:
        print(" All 191 features are already completely calculated. Nothing left to process!")
        exit()

    # asynchronous process worker submission layer
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_single_feature_gpglmm, ffile, gldf_shared, TARGET_NTREES): ffile
            for ffile in files_to_process
        }

        # dictionary loop updates and saves progress incrementally
        for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Running 100-Tree Production", unit="feature"):
            try:
                univ_key, stats_row = future.result()
                xdict[univ_key] = stats_row

                # intermediate save progress
                save_checkpoint(xdict, output_excel)

            except Exception as crash_err:
                print(f"\n Worker process crashed on an active thread: {str(crash_err)}")
                continue

    print(f"\n Complete 191-Feature analysis finalized! Core metrics log saved to: {output_excel}")
