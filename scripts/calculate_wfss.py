import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import argparse


def get_frame_labels(scan_id, labels_df):
    if ".mha" not in scan_id:
        scan_id += ".mha"
    scan_df = labels_df[labels_df['Filename'] == scan_id]
    labels_ids = scan_df["Label"].to_list()
    mapping = {0: "irrelevant", 1: "optimal", 2: "suboptimal"}
    labels = [mapping.get(v, v) for v in labels_ids]
    return labels


def get_frame_labels_from_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    labels = []
    for frame in mask_array:
        unique = np.unique(frame)
        if 1 in unique:
            labels.append("optimal")
        elif 2 in unique:
            labels.append("suboptimal")
        else:
            labels.append("irrelevant")
    return labels


def calc_scan_wfss(scan_id, selected, labels_df = None, mask_dir=None):

    if labels_df is not None:
        frame_labels = get_frame_labels(scan_id, labels_df)
    elif mask_dir is not None:
        mask_path = os.path.join(mask_dir, scan_id)
        if not os.path.exists(mask_path):
            return
        frame_labels = get_frame_labels_from_mask(mask_path)
    else:
        frame_labels = None
    if selected == -1 or selected >= len(frame_labels):
        return 0.0
    label = frame_labels[selected]
    has_optimal = "optimal" in frame_labels
    if label == "optimal":
        score = 1.0
    elif label == "suboptimal" and (has_optimal is False):
        score = 1.0
    elif label == "suboptimal" and has_optimal:
        score = 0.6
    else:
        score = 0.0

    return score


def calculate_wfss(predictions_csv, mask_dir=None, labels_df=None):
    pred_df = pd.read_csv(predictions_csv)
    scores = []
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        scan_id = row["scan"]
        selected = int(row["best_frame"])

        score = calc_scan_wfss(scan_id, selected, labels_df = labels_df, mask_dir=None)
        scores.append(score)

    wfss_score = np.mean(scores)
    print(f"WFSS: {wfss_score:.4f}")
    return wfss_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Weighted Frame Selection Score (WFSS)")
    parser.add_argument("--predictions", type=str, required=True, help="Path to CSV file with predictions")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to directory with masks")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels csv file")
    args = parser.parse_args()

    eval_df = pd.read_csv(args.labels_path)
    calculate_wfss(args.predictions, args.mask_dir,eval_df)
