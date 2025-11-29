import os
import glob
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision import transforms

from part1_frame_classification.src.utils import load_mha_frames
from part1_frame_classification.scripts.calculate_wfss import calc_scan_wfss
from part1_frame_classification.scripts.train_classification import build_model


# -------------------------------------------------
#  Helpers
# -------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_optimal_suboptimal(scan_id: str, labels_df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """
    Extract lists of optimal and suboptimal frame indices for a given scan.
    labels_df must have columns: 'Filename' (e.g. "xxx.mha"), 'Frame', 'Label'
    where Label=1 -> optimal, Label=2 -> suboptimal.
    """
    scan_name = scan_id + ".mha"
    scan_df = labels_df[labels_df["Filename"] == scan_name]

    optimal_slices = scan_df.loc[scan_df["Label"] == 1, "Frame"].to_list()
    suboptimal_slices = scan_df.loc[scan_df["Label"] == 2, "Frame"].to_list()

    return optimal_slices, suboptimal_slices


def read_test_scans(val_scans_path: str):
    """
    Given cross_valid_folds/<k>/test with:
        irrelevant/, optimal/, suboptimal/
    return a set of scan base names (without frame suffix).
    """
    frames = []
    for cls_name in ["irrelevant", "optimal", "suboptimal"]:
        cls_dir = os.path.join(val_scans_path, cls_name)
        if os.path.exists(cls_dir):
            frames.extend(os.listdir(cls_dir))

    scans_set = set()
    for frame_name in frames:
        # frames like: <scan>_<frame>.png
        name = frame_name.split("_")[0]
        scans_set.add(name)
    return scans_set


@torch.no_grad()
def score_frames(model: torch.nn.Module,
                 frames_tensor: torch.Tensor,
                 device: torch.device) -> np.ndarray:
    """
    Given a model and a [N, C, H, W] tensor of frames,
    return softmax probabilities as a numpy array [N, num_classes].
    """
    model.eval()
    frames_tensor = frames_tensor.to(device)
    logits = model(frames_tensor)
    probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def process_scans(
    scan_dir: str,
    model: torch.nn.Module,
    transform,
    val_scan_names,
    labels_df: pd.DataFrame,
    threshold: float = 0.0,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    For each .mha scan in scan_dir that belongs to val_scan_names:
    - Score all frames
    - Choose best frame by optimal probability (class index 1)
    - Compute WFSS via calc_scan_wfss
    - Store per-scan info in a DataFrame.
    """

    results = []
    all_scan_paths = sorted(glob.glob(os.path.join(scan_dir, "*.mha")))

    for path in tqdm(all_scan_paths, desc="Processing validation/test scans"):
        scan_id = os.path.basename(path).replace(".mha", "")
        if scan_id not in val_scan_names:
            continue

        try:
            print(f"processing scan {scan_id}")
            frames_tensor = load_mha_frames(path, transform)  # [N,1,H,W]
            if frames_tensor is None or frames_tensor.size(0) == 0:
                print(f"  ⚠ no frames loaded for {scan_id}, skipping.")
                continue

            probs = score_frames(model, frames_tensor, device)  # [N,3]
            # class 1 = optimal (0=irrelevant, 1=optimal, 2=suboptimal)
            scores = probs[:, 1]
            max_score = float(np.max(scores))
            best_idx = int(np.argmax(scores)) if max_score >= threshold else -1

            optimal_frames, suboptimal_frames = get_optimal_suboptimal(scan_id, labels_df)
            wfss_score = calc_scan_wfss(scan_id, best_idx, labels_df)
            print(f"  wfss_score: {wfss_score:.4f}")

            results.append(
                {
                    "scan": scan_id + ".mha",
                    "best_frame": best_idx,
                    "score": max_score,
                    "optimal_frames": optimal_frames,
                    "suboptimal_frames": suboptimal_frames,
                    "wfss": wfss_score,
                }
            )

        except Exception as e:
            print(f"Error processing {scan_id}: {e}")

    return pd.DataFrame(results)


# -------------------------------------------------
#  Main
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate frame selection with WFSS metrics")
    parser.add_argument("--scan_dir", type=str, required=True, help="Path to .mha scans directory")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt)")
    parser.add_argument(
        "--test_scans_path",
        type=str,
        required=True,
        help="Path to cross_valid_folds/<k>/test directory",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default="true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of classes for the classifier (default 3)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "convnext", "densenet", "resnet"],
        help="Backbone architecture (must match training)",
    )
    args = parser.parse_args()

    # ---- labels ----
    labels_df = pd.read_csv(args.labels_path)

    # ---- device ----
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---- model ----
    print(f"[info] loading model {args.model} with arch={args.arch}, num_classes={args.num_classes}")
    model = build_model(args.arch, num_classes=args.num_classes, device=device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- which scans to evaluate ----
    val_scan_names = read_test_scans(args.test_scans_path)

    # ---- transform (match 3-class eval stats) ----
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4589225], [0.15043989]),
        ]
    )

    # ---- run per-scan evaluation ----
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_results = process_scans(
        args.scan_dir,
        model,
        transform,
        val_scan_names,
        labels_df,
        threshold=0.0,
        device=device,
    )

    # Save per-scan predictions
    df_results.to_csv(args.output, index=False)
    print(f"✅ Per-scan predictions saved to {args.output}")

    # ---- compute per-fold metrics: mean WFSS & optimal_hit_rate ----
    if df_results.empty:
        print("⚠ No scan results; skipping metrics aggregation.")
    else:
        # mean WFSS across all evaluated scans
        mean_wfss = float(df_results["wfss"].mean())

        # optimal_hit_rate: among scans that actually have optimal frames
        hits = 0
        total_with_opt = 0
        for _, row in df_results.iterrows():
            opt_frames = row["optimal_frames"]
            best_idx = row["best_frame"]
            if len(opt_frames) > 0:
                total_with_opt += 1
                if best_idx in opt_frames:
                    hits += 1

        optimal_hit_rate = float(hits / total_with_opt) if total_with_opt > 0 else 0.0
        num_scans = int(len(df_results))

        metrics = pd.DataFrame(
            [
                {
                    "mean_wfss": mean_wfss,
                    "optimal_hit_rate": optimal_hit_rate,
                    "num_scans": num_scans,
                }
            ]
        )

        metrics_path = os.path.splitext(args.output)[0] + "_metrics.csv"
        metrics.to_csv(metrics_path, index=False)

        print("\n=== Fold metrics ===")
        print(f"mean_wfss       = {mean_wfss:.4f}")
        print(f"optimal_hit_rate= {optimal_hit_rate:.4f}")
        print(f"num_scans       = {num_scans}")

        print(f"\n✅ Metrics saved to {metrics_path}")
