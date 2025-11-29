import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
import torch.nn as nn
import pandas as pd
import argparse
from part1_frame_classification.src.models import (get_finetune_efficientnetv2_s_gray,
    get_finetune_densenet121_gray, get_finetune_resnet_model, get_finetune_convnext_small )
from part1_frame_classification.src.utils import (
    mixup_data, mixup_criterion, evaluate,
    get_create_model_dir, ParamsReadWrite,
    FocalLoss, LDAMLoss, FocalWFSSLoss, CEWFSSLoss,str2bool
)

# ----------------------------
# Transforms (1-channel + grayscale stats)
# ----------------------------
train_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.4589225], [0.15043989]),
])

eval_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4589225], [0.15043989]),
])



def get_dataloaders(data_dir: str, batch_size: int):
    train_ds = ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds   = ImageFolder(os.path.join(data_dir, 'val'),   transform=eval_tf)
    test_ds  = ImageFolder(os.path.join(data_dir, 'test'),  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

def build_model(model_name: str, num_classes: int, device):
    model_name = model_name.lower()

    if model_name == "efficientnet":
        return get_finetune_efficientnetv2_s_gray(
            num_classes=num_classes, pretrained=True, device=device
        )

    elif model_name == "convnext":
        return get_finetune_convnext_small(
            num_classes=num_classes, pretrained=True, device=device
        )

    elif model_name == "densenet":
        return get_finetune_densenet121_gray(
            num_classes=num_classes, pretrained=True, device=device
        )

    elif model_name == "resnet":
        return get_finetune_resnet_model(
            num_classes=num_classes, pretrained=True, grayscale=True, device=device
        )

    else:
        raise ValueError(f" Unknown model: {model_name}. "
                         f"Choose: efficientnet / convnext / densenet / resnet")
        
def build_criterion(args, num_classes: int, cls_num_list=None):
    """
    Create the requested loss function based on args.loss_type.
    """
    ls = max(0.0, min(0.99, args.label_smoothing))
    loss_name = args.loss_type.lower()

    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)

    elif loss_name == "focal":
        return FocalLoss()

    elif loss_name == "ldam":
        if cls_num_list is None:
            raise ValueError("cls_num_list is required for LDAM loss.")
        return LDAMLoss(cls_num_list=cls_num_list)

    elif loss_name == "focal_wfss":
        return FocalWFSSLoss(
            idx_opt=args.idx_opt,
            idx_sub=args.idx_sub,
            subopt_score=args.wfss_sub_score,
            wfss_weight=args.wfss_lambda,
        )

    elif loss_name == "ce_wfss":
        return CEWFSSLoss(
            idx_opt=args.idx_opt,
            idx_sub=args.idx_sub,
            subopt_score=args.wfss_sub_score,
            ce_weight=1.0,
            wfss_weight=args.wfss_lambda,
            label_smoothing=ls,
        )

    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")
    
    
def train(model, train_loader, val_loader, optimizer, criterion,
          epochs=50, device="cuda", model_save_path='model.pt',
          metrics_path='train_metrics.csv', patience=20, min_epoch=20,
          apply_mixup=False):

    scaler = GradScaler('cuda' if device=='cuda' else 'cpu')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.1
    )

    best_val_acc = 0.0
    counter = 0

    df = pd.DataFrame({"epoch": pd.Series(dtype="int"),
                       "train_loss": pd.Series(dtype="float32"),
                       "train_acc": pd.Series(dtype="float32"),
                       "val_acc": pd.Series(dtype="float32")})
    df.to_csv(metrics_path, index=False)

    torch.backends.cudnn.benchmark = True

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=(device=='cuda')):
                if apply_mixup:
                    mixed_images, targets_a, targets_b, lam = mixup_data(images, labels)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_acc)

        # early-stopping
        if epoch >= min_epoch and val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f" Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")
        elif epoch < min_epoch:
            pass
        else:
            counter += 1
            if counter >= patience:
                print(" Early stopping triggered")
                break

        # log epoch
        metrics_df = pd.read_csv(metrics_path)
        row = {"epoch": epoch, "train_loss": train_loss,
               "train_acc": train_acc, "val_acc": val_acc}
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
        metrics_df.to_csv(metrics_path, index=False)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-V2-S (gray, 3-class) for Acouslic-AI")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset with train/val/test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_dir", type=str, default="./part1_frame_classification/output/network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--min_epoch", type=int, default=10, help="Min epoch to allow saving")
    parser.add_argument("--apply_mixup", type=str2bool, default=False, help="Apply mixup")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="CE label smoothing (0..1)")
    parser.add_argument("--model", type=str, default="efficientnet",
                    choices=["efficientnet", "convnext", "densenet", "resnet"],
                    help="Which grayscale model to train")
    parser.add_argument("--loss_type", type=str, default="ce",
                    choices=["ce", "focal", "ldam", "ce_wfss", "focal_wfss"],
                    help="Loss function to use")
    parser.add_argument("--idx_opt", type=int, default=2,
                        help="Class index for 'optimal' frames (for WFSS-based losses)")
    parser.add_argument("--idx_sub", type=int, default=1,
                        help="Class index for 'suboptimal' frames (for WFSS-based losses)")
    parser.add_argument("--wfss_sub_score", type=float, default=0.6,
                        help="Score assigned to 'suboptimal' class in WFSS surrogate")
    parser.add_argument("--wfss_lambda", type=float, default=0.3,
                        help="Weight of WFSS term in CE+WFSS loss")
    args = parser.parse_args()

    # Prepare output dirs & config
    model_dir = get_create_model_dir(args.log_dir)
    model_save_path = os.path.join(model_dir, 'model.pt')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    out_cfg_path = os.path.join(model_dir, 'config.json')
    ParamsReadWrite.write_config(
    out_cfg_path, args.data_dir, args.epochs, args.batch_size, args.lr,
    args.weight_decay, args.patience, args.min_epoch, args.apply_mixup,
    3, f"{args.model}_CE+LS")
    metrics_path = os.path.join(model_dir, 'train_metrics.csv')

    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    
    num_classes = 3

    # Class counts for LDAM (only needed if loss_type == 'ldam')
    cls_num_list = None
    if args.loss_type == "ldam":
        # ImageFolder exposes labels via .targets
        targets = train_loader.dataset.targets
        t = torch.tensor(targets, dtype=torch.long)
        cls_num_list = [(t == c).sum().item() for c in range(num_classes)]
        print(f" LDAM cls_num_list: {cls_num_list}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(args.model, num_classes=3, device=device)
    criterion = build_criterion(args, num_classes=num_classes, cls_num_list=cls_num_list)
    criterion.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    print(f" Training started with config: {args}")
    
    train(model, train_loader, val_loader, optimizer=optimizer, criterion=criterion,
          epochs=args.epochs, model_save_path=model_save_path, metrics_path=metrics_path,
          patience=args.patience, min_epoch=args.min_epoch, apply_mixup=args.apply_mixup,
          device=device)

    # Evaluate on test set
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f" Test Accuracy: {test_acc:.4f}")
