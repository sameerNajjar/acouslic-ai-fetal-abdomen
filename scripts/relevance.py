import os, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.amp import autocast, GradScaler
from part1_frame_classification.scripts.train_classification import build_model
from part1_frame_classification.src.utils import evaluate, ClassBalancedFocalLoss ,mixup_data, mixup_criterion,str2bool


@torch.no_grad()
def find_temperature(model, loader, device):
    model.eval()
    temps = torch.linspace(0.5, 3.0, steps=26, device=device)
    best_T, best_nll = 1.0, float("inf")
    for T in temps:
        nll, total = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x) / T
            nll += F.cross_entropy(logits, y, reduction="sum").item()
            total += y.size(0)
        cur = nll / max(1, total)
        if cur < best_nll:
            best_nll, best_T = cur, float(T)
    return best_T

def safe_pil_loader(path: str):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as e:
        print(f"[WARN] bad image skipped: {path} ({e})")
        return None


REL_MAP = {"irrelevant": 0, "optimal": 1, "suboptimal": 1}

def make_relevance_dataset(root, split, transform):
    ds = ImageFolder(os.path.join(root, split), transform=transform, loader=safe_pil_loader)

    # old id -> new id
    old2new = {ds.class_to_idx[k]: v for k, v in REL_MAP.items() if k in ds.class_to_idx}

    # rewrite samples/targets to reflect remap
    new_samples, new_targets = [], []
    for p, y in ds.samples:
        if y in old2new:
            ny = old2new[y]
            new_samples.append((p, ny))
            new_targets.append(ny)
        # else: drop 

    ds.samples = new_samples
    ds.targets = new_targets
    ds.classes = ["irrelevant", "relevant"]
    ds.class_to_idx = {"irrelevant": 0, "relevant": 1}
    return ds


def get_relevance_dataloaders(base_path, batch_size):
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.4589225], [0.15043989]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4589225], [0.15043989]),
        ]
    )

    train_ds = make_relevance_dataset(base_path, "train", train_tf)
    val_ds = make_relevance_dataset(base_path, "val", eval_tf)
    test_ds = make_relevance_dataset(base_path, "test", eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    targets = np.array(train_ds.targets)
    neg = int((targets == 0).sum())
    pos = int((targets == 1).sum())
    return train_loader, val_loader, test_loader, [neg, pos]



def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs=20,
    device="cuda",
    model_save_path="model.pt",
    metrics_path="train_metrics.csv",
    patience=8,
    min_epoch=5,
    apply_mixup=False,
    mixup_alpha=0.4,
):
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler("cuda" if device.startswith("cuda") else "cpu")
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.1
    )
    best_val, wait = 0.0, 0

    # init metrics csv
    pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_acc"]).to_csv(
        metrics_path, index=False
    )

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=device.startswith("cuda")):
                if apply_mixup:
                    mixed, ya, yb, lam = mixup_data(images, labels, alpha=mixup_alpha)
                    outputs = model(mixed)
                    loss = mixup_criterion(criterion, outputs, ya, yb, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | loss {train_loss:.4f} | "
            f"train {train_acc:.4f} | val {val_acc:.4f}"
        )

        # log
        prev = pd.read_csv(metrics_path)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(
            metrics_path, index=False
        )

        sched.step(val_acc)

        # early stopping
        if epoch + 1 >= min_epoch and val_acc > best_val:
            best_val, wait = val_acc, 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  ✔ saved best @ epoch {epoch+1} (val {val_acc:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print("  ⏹ early stopping")
                break


# ----------------- main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Binary relevance trainer (irrelevant vs relevant)")
    parser.add_argument("--data_dir", default="./acouslic_dataset/cross_valid_folds/0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--apply_mixup", type=str2bool, default=False)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--log_dir", type=str, default="./part1_frame_classification/output/relevance")
    parser.add_argument(
        "--model",
        type=str,
        default="convnext",
        choices=["efficientnet", "convnext", "densenet", "resnet"],
        help="Backbone architecture for relevance model",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.log_dir, exist_ok=True)
    model_dir = args.log_dir
    model_path = os.path.join(model_dir, "relevance_model.pt")
    metrics_path = os.path.join(model_dir, "relevance_metrics.csv")

    # data
    train_loader, val_loader, test_loader, counts = get_relevance_dataloaders(
        args.data_dir, args.batch_size
    )
    print(f"[info] train counts (irrelevant,relevant) = {counts}")

    # model & loss
    model = build_model(args.model, num_classes=2, device=device)
    print(f"[info] using model backbone: {args.model}")

    criterion = ClassBalancedFocalLoss(
        class_counts=counts, gamma=2.0, label_smoothing=0.05, beta=0.999, device=device
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # train
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=args.epochs,
        device=device,
        model_save_path=model_path,
        metrics_path=metrics_path,
        patience=8,
        min_epoch=5,
        apply_mixup=args.apply_mixup,
        mixup_alpha=args.mixup_alpha,
    )

    # test + calibration
    model.load_state_dict(torch.load(model_path, map_location=device))
    T_rel = find_temperature(model, val_loader, device)
    with open(os.path.join(model_dir, "relevance_T.txt"), "w") as f:
        f.write(str(T_rel))
    print(f"[calibration] relevance temperature T = {T_rel:.3f}")
    test_acc = evaluate(model, test_loader, device)
    print(f"[TEST] accuracy: {test_acc:.4f}")
