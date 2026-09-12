import os
import sys
import glob
import shutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# optimize for ryzen 9 7950X cores
os.environ["OMP_NUM_THREADS"] = "16"       # lock each active worker process to 16 threads
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

# force Python to check active working folder first
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# import the engine from utils
from utils.gpglmm_engine import process_single_feature_gpglmm

if __name__ == "__main__":
    # destination files
    output_excel = "output/GPGLMM_PRODUCTION_191_100tree.xlsx"
    predictions_folder = "output/model_predictions"

    print(" Loading core Glottolog geographic and language data tables...")
    gldf = pd.read_csv("tlu/Glottolog_Languages.csv")
    gldf['Family_ID'] = gldf['Family_ID'].replace(['NaN', 'nan', 'None', ''], pd.NA)
    gldf['Family_ID'] = gldf['Family_ID'].fillna(gldf['glottocode']).astype(str).str.strip()
    gldf = gldf.dropna(subset=['latitude'])
    gldf_shared = gldf[['glottocode', 'macroarea', 'Family_ID', 'longitude', 'latitude']].copy()

    print(" Dynamically scanning 'tlu/' repository folders for ALL features...")

    # scan all subfolder directories to build the 191 feature queue
    search_pattern = os.path.join("tlu", "*", "*data.txt")
    files_to_process = sorted(glob.glob(search_pattern))

    if not files_to_process:
        print(" Error: Could not locate any raw '*data.txt' feature files inside your 'tlu/' folder structure.")
        exit()

    TARGET_NTREES = 100
    MAX_WORKERS = 2  # balanced worker allocation for 32 threads total (2 x 16 threads)

    print("\n" + "="*75)
    print(" LAUNCHING PRODUCTION 191-FEATURE 100-TREE ANALYSIS PIPELINE")
    print("="*75)
    print(f" Target Queue Depth  : {len(files_to_process)} Total Universals Selected")
    print(f" System Engine Load  : Spinning up {MAX_WORKERS} concurrent processes")
    print(f" Thread Parameters   : 16 CPU cores assigned per active process")
    print(f" Sampling Density    : Running across ALL {TARGET_NTREES} tree matrices per feature")
    print(f" Destination Files   : {output_excel}\n")

    run_pipeline = True
    if os.path.isfile(output_excel):
        response = input(f"⚠️ Warning: '{os.path.basename(output_excel)}' already exists.\nContinuing will overwrite previous results.\nDo you wish to proceed? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Pipeline execution aborted by user choice.")
            run_pipeline = False
            exit()

    xdict = {}

    if run_pipeline:
        # Asynchronous process worker submission layer targeting the full file queue
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(process_single_feature_gpglmm, ffile, gldf_shared, TARGET_NTREES): ffile
                for ffile in files_to_process
            }

            # Tqdm handles terminal progress bars across the full 191 queue
            for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Running 100-Tree Production", unit="feature"):
                try:
                    univ_key, stats_row = future.result()
                    xdict[univ_key] = stats_row
                except Exception as crash_err:
                    print(f"\n💥 Worker process crashed on an active thread: {str(crash_err)}")
                    continue

    # Compile dataset output rows and export spreadsheet results
    if xdict:
        df_final = pd.DataFrame.from_dict(xdict, orient='index')
        ordered_cols = [
            "Status", "Reason", "Total_Languages_Found", "Distinct_Macroareas",
            "Distinct_Families", "DV_Variance", "DV_Mean", "GPGLMM_n_obs",
            "GPGLMM_Param.", "GPGLMM_Std. err.", "GPGLMM_z value", "GPGLMM_P>|z|",
            "GPGLMM_sig"
        ]
        available_cols = [c for c in ordered_cols if c in df_final.columns]
        df_final = df_final[available_cols]
        df_final.to_excel(output_excel, index_label="Feature_ID")

        print(f"\n 191-Feature Production Run complete! Core metrics log saved to: {output_excel}")
