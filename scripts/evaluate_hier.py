import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision import transforms

from part1_frame_classification.scripts.train_classification import build_model
from part1_frame_classification.src.utils import load_mha_frames, str2bool


# ----------------- helpers -----------------
def read_test_scans(test_scans_path: str):
    """
    Collect scan IDs that appear in the test split frames.
    We assume filenames of the form: <scanid>_<frame>.png
    under folders: irrelevant / optimal / suboptimal.
    """
    frames = []
    for cls in ["irrelevant", "optimal", "suboptimal"]:
        cls_dir = os.path.join(test_scans_path, cls)
        if os.path.isdir(cls_dir):
            frames.extend(os.listdir(cls_dir))

    scan_ids = set()
    for frame_name in frames:
        if "_" in frame_name:
            scan_ids.add(frame_name.split("_")[0])
    return scan_ids


def get_frame_labels(scan_id: str, labels_df: pd.DataFrame):
    """
    Extract ground-truth optimal / suboptimal frame indices for a scan.
    """
    scan_df = labels_df[labels_df["Filename"] == f"{scan_id}.mha"]
    optimal_frames = scan_df.loc[scan_df["Label"] == 1, "Frame"].to_list()
    suboptimal_frames = scan_df.loc[scan_df["Label"] == 2, "Frame"].to_list()
    return optimal_frames, suboptimal_frames


def compute_wfss(best_idx: int,
                 optimal_frames,
                 suboptimal_frames,
                 sub_score: float = 0.6) -> float:
    """
    Compute WFSS for a single scan, given:
      - best_idx: predicted best frame index (or -1 if none)
      - optimal_frames: list of frame indices with Label=1
      - suboptimal_frames: list of frame indices with Label=2
      - sub_score: WFSS reward for picking a suboptimal frame
                   when an optimal exists.
    """
    if best_idx < 0:
        return 0.0

    has_opt = len(optimal_frames) > 0

    if has_opt:
        if best_idx in optimal_frames:
            return 1.0
        if best_idx in suboptimal_frames:
            return float(sub_score)
        return 0.0
    else:
        # no optimal frames in this scan
        if best_idx in suboptimal_frames:
            return 1.0
        return 0.0


def maybe_load_temperature(default_T: float, model_path: str, filename: str) -> float:
    """
    Try to load a temperature scalar from model_dir/filename.
    If not found or parsing fails, return default_T.
    """
    model_dir = os.path.dirname(model_path)
    T_path = os.path.join(model_dir, filename)
    if os.path.exists(T_path):
        try:
            with open(T_path, "r") as f:
                return float(f.read().strip())
        except Exception:
            pass
    return default_T


@torch.no_grad()
def score_frames(model, frames_tensor, device, batch_size=64, temperature=1.0):
    """
    Run a model over all frames in a scan, returning softmax probabilities.
    frames_tensor: [N, C, H, W] on CPU
    returns: np.array of shape [N, num_classes]
    """
    model.eval()
    N = frames_tensor.size(0)
    probs = []

    for i in range(0, N, batch_size):
        batch = frames_tensor[i:i + batch_size].to(device, non_blocking=True)
        logits = model(batch) / float(temperature)
        p = torch.softmax(logits, dim=1)
        probs.append(p.cpu())

    return torch.cat(probs, dim=0).numpy()


# ----------------- main evaluation logic -----------------
def main():
    parser = argparse.ArgumentParser(
        "Evaluate hierarchical (relevance + quality) frame selection on .mha scans"
    )
    parser.add_argument("--scan_dir", type=str, required=True,
                        help="Path to .mha scans (stacked_fetal_ultrasound)")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Path to labels.csv (all frames)")
    parser.add_argument("--test_scans_path", type=str, required=True,
                        help="Path to fold test frames (irrelevant/optimal/suboptimal folders)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save per-scan predictions CSV")

    parser.add_argument("--relevance_model", type=str, required=True,
                        help="Path to trained relevance model .pt")
    parser.add_argument("--quality_model", type=str, required=True,
                        help="Path to trained quality model .pt")

    parser.add_argument("--arch_rel", type=str, default="convnext",
                        choices=["efficientnet", "convnext", "densenet", "resnet"],
                        help="Backbone architecture for relevance model")
    parser.add_argument("--arch_qual", type=str, default="densenet",
                        choices=["efficientnet", "convnext", "densenet", "resnet"],
                        help="Backbone architecture for quality model")

    parser.add_argument("--rel_threshold", type=float, default=0.0,
                        help="Minimum P(relevant) to keep a frame for quality model")
    parser.add_argument("--sub_score", type=float, default=0.6,
                        help="WFSS reward for selecting suboptimal when optimal exists")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_gpu", type=str2bool, default="true")

    args = parser.parse_args()

    # device
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[info] using device: {device}")

    # labels
    labels_df = pd.read_csv(args.labels_path)

    # which scans belong to this fold?
    test_scan_ids = read_test_scans(args.test_scans_path)
    print(f"[info] #test scans in this fold: {len(test_scan_ids)}")

    # build models (both binary)
    rel_model = build_model(args.arch_rel, num_classes=2, device=device)
    qual_model = build_model(args.arch_qual, num_classes=2, device=device)

    # load weights
    rel_model.load_state_dict(torch.load(args.relevance_model, map_location=device))
    qual_model.load_state_dict(torch.load(args.quality_model, map_location=device))

    # load (optional) temperatures
    T_rel = maybe_load_temperature(1.0, args.relevance_model, "relevance_T.txt")
    T_qual = maybe_load_temperature(1.0, args.quality_model, "quality_T.txt")
    print(f"[info] using T_rel={T_rel:.3f}, T_qual={T_qual:.3f}")

    # transforms for each model (match training as closely as possible)
    rel_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4589225], [0.15043989]),
    ])

    qual_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.14], [0.15]),
    ])

    # ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # iterate scans
    all_scan_paths = sorted(glob.glob(os.path.join(args.scan_dir, "*.mha")))
    rows = []

    wfss_list = []
    opt_hit_list = []
    scans_with_opt = 0

    for path in tqdm(all_scan_paths, desc="Evaluating scans"):
        scan_id = os.path.basename(path).replace(".mha", "")
        if scan_id not in test_scan_ids:
            continue

        try:
            # load frames with both transforms (order must match)
            frames_rel = load_mha_frames(path, rel_tf)    # [N, 1, H, W]
            frames_qual = load_mha_frames(path, qual_tf)  # [N, 1, H, W]
            N = frames_rel.size(0)

            # --- relevance model ---
            probs_rel = score_frames(rel_model, frames_rel, device,
                                     batch_size=args.batch_size,
                                     temperature=T_rel)
            p_rel = probs_rel[:, 1]   # class 1 = relevant

            # mask of frames considered relevant
            rel_mask = p_rel >= float(args.rel_threshold)
            rel_indices = np.where(rel_mask)[0]

            if rel_indices.size == 0:
                best_idx = -1
                best_score = 0.0
                p_rel_best = 0.0
                p_opt_best = 0.0
            else:
                # --- quality model on relevant frames ---
                frames_qual_rel = frames_qual[rel_mask]  # subset
                probs_qual = score_frames(qual_model, frames_qual_rel, device,
                                          batch_size=args.batch_size,
                                          temperature=T_qual)
                p_opt = probs_qual[:, 1]  # class 1 = optimal

                # combine scores
                combined = p_rel[rel_mask] * p_opt  # [M]
                local_best = int(np.argmax(combined))
                best_idx = int(rel_indices[local_best])
                best_score = float(combined[local_best])
                p_rel_best = float(p_rel[best_idx])
                p_opt_best = float(p_opt[local_best])

            # ground truth lists
            optimal_frames, suboptimal_frames = get_frame_labels(scan_id, labels_df)

            # WFSS
            wfss = compute_wfss(best_idx, optimal_frames, suboptimal_frames,
                                sub_score=args.sub_score)
            wfss_list.append(wfss)

            # optimal-hit rate statistics
            has_opt = len(optimal_frames) > 0
            if has_opt:
                scans_with_opt += 1
                opt_hit = 1.0 if (best_idx in optimal_frames) else 0.0
                opt_hit_list.append(opt_hit)

            rows.append({
                "scan": scan_id + ".mha",
                "best_frame": best_idx,
                "score": best_score,
                "p_relevant": p_rel_best,
                "p_optimal": p_opt_best,
                "wfss": wfss,
                "optimal_frames": optimal_frames,
                "suboptimal_frames": suboptimal_frames,
            })

        except Exception as e:
            print(f"[ERROR] Failed processing {scan_id}: {e}")

    # save per-scan predictions
    df_results = pd.DataFrame(rows)
    df_results.to_csv(args.output, index=False)
    print(f"[info] per-scan predictions saved to: {args.output}")

    # aggregate metrics
    if wfss_list:
        mean_wfss = float(np.mean(wfss_list))
    else:
        mean_wfss = 0.0

    if opt_hit_list and scans_with_opt > 0:
        optimal_hit_rate = float(np.sum(opt_hit_list) / scans_with_opt)
    else:
        optimal_hit_rate = 0.0

    metrics_path = os.path.splitext(args.output)[0] + "_metrics.csv"
    metrics_df = pd.DataFrame([{
        "mean_wfss": mean_wfss,
        "optimal_hit_rate": optimal_hit_rate,
        "num_scans": len(wfss_list),
        "num_scans_with_optimal": scans_with_opt,
    }])
    metrics_df.to_csv(metrics_path, index=False)

    print("\n=== Hierarchical evaluation metrics ===")
    print(f"mean WFSS:           {mean_wfss:.4f}")
    print(f"optimal hit rate:    {optimal_hit_rate:.4f}")
    print(f"#scans (this fold):  {len(wfss_list)}")
    print(f"#scans with optimal: {scans_with_opt}")
    print(f"[info] metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
