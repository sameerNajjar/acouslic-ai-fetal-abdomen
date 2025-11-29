import os, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.amp import autocast, GradScaler

# Keep your existing imports
from part1_frame_classification.scripts.train_classification import build_model
from part1_frame_classification.src.utils import evaluate

# ---------------- Dataset & Transforms ----------------
def safe_pil_loader(path: str):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as e:
        print(f"[WARN] bad image skipped: {path} ({e})")
        return None

def make_quality_dataset(root, split, transform):
    ds = ImageFolder(os.path.join(root, split), transform=transform, loader=safe_pil_loader)
    
    # Ensure mapping is: suboptimal=0, optimal=1
    if "optimal" not in ds.class_to_idx or "suboptimal" not in ds.class_to_idx:
        raise RuntimeError(f"{split} must contain 'optimal' and 'suboptimal' folders.")

    old_opt = ds.class_to_idx["optimal"]
    old_sub = ds.class_to_idx["suboptimal"]
    old2new = {old_sub: 0, old_opt: 1}

    new_samples, new_targets = [], []
    for p, y in ds.samples:
        if y in (old_sub, old_opt):
            ny = old2new[y]
            new_samples.append((p, ny))
            new_targets.append(ny)

    ds.samples = new_samples
    ds.targets = new_targets
    ds.classes = ["suboptimal", "optimal"]
    ds.class_to_idx = {"suboptimal": 0, "optimal": 1}
    return ds

def get_dataloaders(base_path, batch_size):
    # Standard Grayscale stats (approx)
    MEAN, STD = 0.14, 0.15

    # 1. Augmentations (Cleaned up)
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([MEAN], [STD]),
    ])
    
    eval_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([MEAN], [STD]),
    ])

    train_ds = make_quality_dataset(base_path, "train", train_tf)
    val_ds   = make_quality_dataset(base_path, "val",   eval_tf)
    test_ds  = make_quality_dataset(base_path, "test",  eval_tf)

    # Standard shuffle is fine for balanced-ish data
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

# ---------------- Training Logic ----------------
def evaluate_detailed(model, loader, device):
    model.eval()
    correct, total = 0, 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for c in range(2):
                mask = y == c
                class_correct[c] += (pred[mask] == y[mask]).sum().item()
                class_total[c] += mask.sum().item()

    acc = correct / max(1, total)
    class_acc = [class_correct[i] / max(1, class_total[i]) for i in range(2)]
    bal_acc = 0.5 * (class_acc[0] + class_acc[1])
    return acc, class_acc, bal_acc

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path, metrics_path):
    scaler = GradScaler("cuda" if device.startswith("cuda") else "cpu")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_bal = 0.0
    pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_acc", "bal_val"]).to_csv(metrics_path, index=False)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=device.startswith("cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_acc, class_acc, bal_acc = evaluate_detailed(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Bal: {bal_acc:.4f} | Sub: {class_acc[0]:.2f} Opt: {class_acc[1]:.2f}")

        # Save metrics
        prev = pd.read_csv(metrics_path)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc, "bal_val": bal_acc}
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(metrics_path, index=False)

        if bal_acc > best_bal:
            best_bal = bal_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved best (Bal: {best_bal:.4f})")
        
        scheduler.step()

# ----------------- Main -----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("Quality Trainer (Frozen Backbone)")
    p.add_argument("--data_dir", default="./acouslic_dataset/cross_valid_folds/0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", type=str, default="densenet",
                   choices=["efficientnet", "convnext", "densenet", "resnet"])
    p.add_argument("--log_dir", type=str, default="./part1_frame_classification/output/quality")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.log_dir, exist_ok=True)
    model_path = os.path.join(args.log_dir, "quality_model.pt")
    metrics_path = os.path.join(args.log_dir, "quality_metrics.csv")

    tr, va, te = get_dataloaders(args.data_dir, args.batch_size)
    
    # 1. Build Model
    model = build_model(args.model, num_classes=2, device=device)
    print(f"[info] Backbone: {args.model}")

    # 2. FREEZE BACKBONE (Crucial Step)
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Unfreeze Head Only
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name or "fc" in name or "head" in name:
            param.requires_grad = True
            head_params.append(param)
            
    print("[info] Backbone Frozen. Training Classifier Head Only.")

    # 4. Standard Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # 5. Optimizer (Only for head)
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-2)

    # 6. Train
    train(model, tr, va, optimizer, criterion, args.epochs, device, model_path, metrics_path)

    # 7. Test
    print("\n[TESTING]")
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_acc = evaluate(model, te, device)
    print(f"Test Accuracy: {test_acc:.4f}")