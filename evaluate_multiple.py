import os
import subprocess
import pandas as pd


if __name__ == "__main__":
    """
    Run evaluation for multiple folds and aggregate WFSS + optimal_hit_rate.
    """

    # ---- CONFIG ----
    data_path = "./acouslic_dataset/cross_valid_folds"
    model_dir = "./part1_frame_classification/output/network"

    # One entry per fold: model_dir/<run_id>/model.pt
    # Example: [74, 75, 76, 77, 78]
    model_dirnames = [74]

    # Full .mha scans directory
    scan_dir = "./acouslic-ai-train-set/images/stacked_fetal_ultrasound"

    # Original labels file (with all frames)
    labels_path = "./acouslic_dataset/labels.csv"

    # Backbone architecture used in training (must match train script)
    # choices: "efficientnet", "convnext", "densenet", "resnet"
    arch = "efficientnet"

    use_gpu = True
    num_classes = 3

    # ----------------
    # Run eval per fold
    # ----------------
    fold_metrics = []

    for fold_idx, run_id in enumerate(model_dirnames):
        print(f"\n=== Evaluating fold {fold_idx} (run id: {run_id}) ===")

        model_path = os.path.join(model_dir, str(run_id), "model.pt")
        results_dir = os.path.join(model_dir, str(run_id), "results")
        os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(results_dir, "predictions.csv")
        test_scans_path = os.path.join(data_path, str(fold_idx), "test")

        # Build CLI args for evaluate_frame_selection
        args = (
            f'--model "{model_path}" '
            f'--scan_dir "{scan_dir}" '
            f'--output "{results_path}" '
            f'--labels_path "{labels_path}" '
            f'--test_scans_path "{test_scans_path}" '
            f'--num_classes {num_classes} '
            f'--arch {arch} '
        )
        if use_gpu:
            args += "--use_gpu true "
        else:
            args += "--use_gpu false "

        cmd = (
            "python -m part1_frame_classification.scripts.evaluate_frame_selection "
            + args
        )
        print(f"Running: {cmd}")
        subprocess.call(cmd, shell=True)

        # Per-fold metrics file created by the eval script
        metrics_path = os.path.splitext(results_path)[0] + "_metrics.csv"

        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            if not df.empty:
                mean_wfss = float(df.loc[0, "mean_wfss"])
                optimal_hit_rate = float(df.loc[0, "optimal_hit_rate"])
                num_scans = int(df.loc[0, "num_scans"])

                fold_metrics.append(
                    {
                        "fold": fold_idx,
                        "run_id": run_id,
                        "mean_wfss": mean_wfss,
                        "optimal_hit_rate": optimal_hit_rate,
                        "num_scans": num_scans,
                    }
                )
                print(
                    f"Fold {fold_idx}: mean_wfss={mean_wfss:.4f}, "
                    f"optimal_hit_rate={optimal_hit_rate:.4f}, num_scans={num_scans}"
                )
            else:
                print(f"Warning: metrics file {metrics_path} is empty.")
        else:
            print(f"Warning: metrics file not found for fold {fold_idx}: {metrics_path}")

    # -----------------------------
    # Aggregate metrics across folds
    # -----------------------------
    if fold_metrics:
        per_fold_df = pd.DataFrame(fold_metrics)

        avg_mean_wfss = per_fold_df["mean_wfss"].mean()
        avg_optimal_hit_rate = per_fold_df["optimal_hit_rate"].mean()
        total_scans = per_fold_df["num_scans"].sum()
        num_folds = len(per_fold_df)

        print("\n=== Cross-fold aggregate metrics ===")
        print(f"Number of folds:         {num_folds}")
        print(f"Total scans (all folds): {total_scans}")
        print(f"Average mean WFSS:       {avg_mean_wfss:.4f}")
        print(f"Average optimal hit rate:{avg_optimal_hit_rate:.4f}")

        # Save per-fold metrics
        per_fold_out = os.path.join(model_dir, "crossval_per_fold_metrics.csv")
        per_fold_df.to_csv(per_fold_out, index=False)

        # Save aggregate averages
        avg_out = os.path.join(model_dir, "crossval_metrics_avg.csv")
        avg_df = pd.DataFrame(
            [
                {
                    "num_folds": num_folds,
                    "total_scans": total_scans,
                    "avg_mean_wfss": avg_mean_wfss,
                    "avg_optimal_hit_rate": avg_optimal_hit_rate,
                }
            ]
        )
        avg_df.to_csv(avg_out, index=False)

        print(f"\n✅ Per-fold metrics saved to: {per_fold_out}")
        print(f"✅ Average metrics saved to:  {avg_out}")
    else:
        print("\n⚠ No per-fold metrics found; nothing to aggregate.")
