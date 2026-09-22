"""
utils/check_datasets.py: Verification script checking the structural integrity
of raw linguistic data tables and compressed phylogenetic trees.
"""
import os
import gzip

def check_linguistic_datasets():
    # Relative path targeting the sister folder from inside the utils directory
    results_dir = "../tlu"

    if not os.path.isdir(results_dir):
        print(f"Error: Cannot find directory '{results_dir}'.")
        return

    # Isolate valid feature subdirectories, skipping summary assets
    subfolders = sorted([
        f for f in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, f)) and f != "BT_results_summary"
    ])

    print(f"Auditing {len(subfolders)} feature subdirectories for structural integrity...\n")
    print(f"{'Feature Folder':<20} | {'Languages':<10} | {'Features':<10} | {'Trees Found':<12}")
    print("-" * 62)

    total_valid = 0

    for folder in subfolders:
        folder_path = os.path.join(results_dir, folder)
        bt_path = os.path.join(folder_path, "BT_data.txt")
        tree_path = os.path.join(folder_path, "pruned_tree.trees.gz")

        languages_count = 0
        features_count = 0
        trees_count = 0
        issue_found = False

        # Ingest and check formatting boundaries in the raw feature data text
        if os.path.exists(bt_path):
            try:
                with open(bt_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

                if lines:
                    languages_count = len(lines)

                    # Read matrix column distribution to calculate feature metrics
                    sample_cols = lines[0].split()
                    if len(sample_cols) > 1:
                        # Index 0 holds the language identifier; remaining columns denote linguistic features
                        features_count = len(sample_cols) - 1
            except Exception as e:
                print(f"Warning: Error reading {bt_path}: {e}")
                issue_found = True
        else:
            issue_found = True

        # Count tree iterations inside the compressed Nexus file archive
        if os.path.exists(tree_path):
            try:
                with gzip.open(tree_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().lower().startswith("tree "):
                            trees_count += 1
            except Exception as e:
                print(f"Warning: Error reading compressed file {tree_path}: {e}")
                issue_found = True
        else:
            issue_found = True

        folder_display = folder if len(folder) <= 20 else folder[:17] + "..."
        print(f"{folder_display:<20} | {languages_count:<10} | {features_count:<10} | {trees_count:<12}")

        if not issue_found and languages_count > 0 and (trees_count == 100 or trees_count == 1000):
            total_valid += 1

    print("-" * 62)
    print(f"Audit Complete: {total_valid}/{len(subfolders)} feature datasets are structurally intact.")

if __name__ == "__main__":
    check_linguistic_datasets()
