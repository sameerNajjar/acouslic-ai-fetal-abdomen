import subprocess
import os


if __name__ == "__main__":
    folds = 1
    data_path = "./acouslic_dataset/cross_valid_folds"
    batch_size = 64
    lr = 3e-4
    apply_mixup = False
    model = "efficientnet"
    loss_type = "wfss"

    # WFSS-related (for wfss / ce_wfss losses)
    wfss_weight = 0.3        
    subopt_score = 0.6       
    idx_sub = 1              
    idx_opt = 2             

    # label smoothing for CE
    label_smoothing = 0.1

    for fold in range(folds):
        print("training fold " + str(fold))
        data_dir = os.path.join(data_path, str(fold))

        args = (
            "python -m part1_frame_classification.scripts.train_classification "
            f"--data_dir {data_dir} "
            f"--batch_size {batch_size} "
            f"--lr {lr} "
            f"--apply_mixup {apply_mixup} "
            f"--model {model} "
            f"--loss_type {loss_type} "
            f"--idx_opt {idx_opt} "
            f"--idx_sub {idx_sub} "
            f"--wfss_sub_score {subopt_score} "
            f"--wfss_lambda {wfss_weight} "
            f"--label_smoothing {label_smoothing}"
        )
        subprocess.call(args, shell=True)
