import pandas as pd
import numpy as np
import os
import argparse

def prepare_balanced_dataset(input_csv, output_csv, suboptimal_limit=10):
    # Check if output file already exists and is not empty
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        if not existing_df.empty:
            print(f" File already exists with {len(existing_df)} rows. Skipping generation.")
            return

    # Load labeled frame CSV
    df = pd.read_csv(input_csv)

    # Validate required columns
    required_cols = {"Filename", "Frame", "Label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    balanced_rows = []

    # Group by scan
    for scan_name, group in df.groupby("Filename"):
        optimal = group[group["Label"] == 1]
        suboptimal = group[group["Label"] == 2]
        irrelevant = group[group["Label"] == 0]

        if len(optimal) == 0 and len(suboptimal) == 0:
            continue  # skip scans with no good frames

        # Sample suboptimal
        suboptimal_sample = suboptimal.sample(n=min(len(suboptimal), suboptimal_limit), random_state=42)

        # Combine good frames
        good_frames = pd.concat([optimal, suboptimal_sample])

        # Sample irrelevant frames equal to good frames
        irrelevant_sample = irrelevant.sample(n=min(len(irrelevant), len(good_frames)), random_state=42)

        # Combine and store
        balanced_rows.append(pd.concat([good_frames, irrelevant_sample]))

    # Combine everything and shuffle
    balanced_df = pd.concat(balanced_rows).sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to new CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    balanced_df.to_csv(output_csv, index=False)
    print(f" Saved reduced 3-class dataset with {len(balanced_df)} frames at: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a balanced 3-class dataset for Acouslic-AI")
    parser.add_argument("--input", type=str, required=True, help="Path to input labeled CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save balanced CSV fil e")
    parser.add_argument("--suboptimal_limit", type=int, default=10, help="Maximum number of suboptimal frames per scan")
    args = parser.parse_args()

    prepare_balanced_dataset(args.input, args.output, args.suboptimal_limit)
