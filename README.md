# Acoustic AI Fetal Abdomen – Frame Classification

A deep learning pipeline for intelligent **frame selection** in fetal abdomen ultrasound scans. The system identifies the optimal frames from 3D ultrasound volumes to maximize clinical utility.

---

##  Overview

This repository implements two complementary approaches for frame classification:

### 1. **3-Class Frame Classifier**
Direct classification into three categories:
- `irrelevant` – Non-diagnostic frames
- `suboptimal` – Diagnostic but not ideal
- `optimal` – Clinically ideal frames

### 2. **Hierarchical Relevance + Quality Model**
Two-stage pipeline:
- **Stage 1 (Relevance)**: Filter out irrelevant frames
- **Stage 2 (Quality)**: Rank remaining frames by quality

Both approaches aim to select the single best frame per 3D scan, maximizing the **Weighted Frame Selection Score (WFSS)** and optimal hit rate.

---

##  Project Structure

```text
acouslic-ai-fetal-abdomen/
├── src/
│   ├── models.py          # Grayscale CNN architectures
│   └── utils.py           # Loss functions, metrics, augmentation
├── scripts/
│   ├── train_classification.py        # 3-class trainer
│   ├── relevance.py                   # Relevance model trainer
│   ├── quality.py                     # Quality model trainer
│   ├── evaluate_frame_selection.py    # Scan-level evaluation
│   ├── calculate_wfss.py              # WFSS computation
│   ├── prepare_balanced_dataset.py    # Dataset balancing
│   └── prepare_csv.py                 # Label file creation
├── train_multiple.py                  # Batch training script
├── evaluate_multiple.py               # Batch evaluation script
├── acouslic_dataset/                  # Training data & labels
└── acouslic-ai-train-set/             # Raw 3D ultrasound volumes
```

---

## Data Organization

### Raw Data
```text
acouslic-ai-train-set/
└── images/
    └── stacked_fetal_ultrasound/
        ├── scan001.mha
        ├── scan002.mha
        └── ...
```

### Processed Dataset
```text
acouslic_dataset/
├── labels.csv
├── labels_balanced.csv
└── cross_valid_folds/
    ├── 0/
    │   ├── data_split/
    │   │   ├── test_ids.txt
    │   │   ├── training_ids.txt
    │   │   └── valid_ids.txt
    │   ├── train/
    │   │   ├── irrelevant/
    │   │   ├── suboptimal/
    │   │   └── optimal/
    │   ├── val/
    │   └── test/
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

---

# Getting Started

## 1. Prepare Balanced Dataset

```bash
python scripts/prepare_balanced_dataset.py \
  --input ./acouslic_dataset/labels.csv \
  --output ./acouslic_dataset/balanced_labels.csv \
  --suboptimal_limit 10
```

## 2. Train 3-Class Classifier

#### Available Models
- `efficientnet` – EfficientNetV2-S (recommended)
- `convnext` – ConvNeXt-Small
- `densenet` – DenseNet121
- `resnet` – ResNet50

#### Available Loss Functions
- `ce` – Cross-entropy with label smoothing
- `focal` – Focal loss
- `ldam` – Label-Distribution-Aware Margin loss
- `ce_wfss` – Cross-entropy + WFSS surrogate (recommended)
- `focal_wfss` – Focal loss + WFSS surrogate

**WFSS-aware losses** encourage scan-level optimization, improving the selection of optimal frames.

#### Training Command

```bash
python scripts/train_classification.py \
  --data_dir ./acouslic_dataset/cross_valid_folds/0 \
  --epochs 30 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --apply_mixup false \
  --label_smoothing 0.1 \
  --model efficientnet \
  --loss_type ce_wfss \
  --idx_opt 2 \
  --idx_sub 1 \
  --wfss_sub_score 0.6 \
  --wfss_lambda 0.3
```

**Output:** Creates experiment folder under `./network/` containing:
- `model.pt` – Best validation checkpoint
- `config.json` – Training configuration
- `train_metrics.csv` – Per-epoch metrics

#### Batch Training

```bash
python train_multiple.py
```

Sweeps over multiple configurations (folds, backbones, loss types).


### Evaluation



```bash
python scripts/evaluate_frame_selection.py \
  --model ./network/87/model.pt \
  --scan_dir ./acouslic-ai-train-set/images/stacked_fetal_ultrasound \
  --labels_path ./acouslic_dataset/labels.csv \
  --output ./network/87/results/predictions.csv \
  --test_scans_path ./acouslic_dataset/cross_valid_folds/0/test \
  --num_classes 3 \
  --arch efficientnet \
  --use_gpu true
```

**Output Files:**
- `predictions.csv` – Per-scan results with selected frames and WFSS
- `predictions_metrics.csv` – Aggregate metrics:
  - `mean_wfss` – Average WFSS across scans
  - `optimal_hit_rate` – Fraction of optimal frames selected
  - `num_scans` – Total scans evaluated

#### Batch Evaluation

```bash
python evaluate_multiple.py
```

---

##  WFSS

**Weighted Frame Selection Score (WFSS)** measures scan-level frame selection quality:

| Scenario | WFSS |
|----------|------|
| Scan has optimal frames → select optimal | **1.0** ✓ |
| Scan has no optimal frames → select suboptimal | **1.0** ✓ |
| Scan has optimal frames → select suboptimal | **0.6** ~ |
| Select irrelevant frame | **0.0** ✗ |

Higher WFSS indicates better clinical utility.

### Recalculate WFSS

```bash
python scripts/calculate_wfss.py \
  --predictions ./network/87/results/predictions.csv \
  --labels ./acouslic_dataset/labels.csv
```

---

# 3. Hierarchical Model

### Architecture

```
All Frames
    ↓
[Relevance Model] → Filter irrelevant
    ↓
Relevant Frames
    ↓
[Quality Model] → Rank by optimality
    ↓
Best Frame
```

### 1. Train Relevance Model

Binary classification: `irrelevant` vs `relevant` (suboptimal + optimal)

```bash
python scripts/relevance.py \
  --data_dir ./acouslic_dataset/cross_valid_folds/0 \
  --epochs 10 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --apply_mixup false \
  --log_dir ./relevance_output
```

**Output:**
- `relevance_model.pt`
- `relevance_metrics.csv`
- this model easly achieves val_acc>0.95

### 2. Train Quality Model

Binary classification: `suboptimal` vs `optimal` (trained only on relevant frames)

```bash
python scripts/quality.py \
  --data_dir ./acouslic_dataset/cross_valid_folds/0 \
  --epochs 30 \
  --batch_size 64 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --apply_mixup false \
  --model efficientnet \
  --loss_type weighted_ce \
  --log_dir ./quality_output
```

**Output:**
- `quality_model.pt`
- `quality_metrics.csv`

### 3. Evaluate Hierarchical Pipeline

Use the same evaluation script, which automatically detects the two-stage architecture.
```bash
python -m part1_frame_classification.scripts.evaluate_hier \
  --scan_dir ./acouslic-ai-train-set/images/stacked_fetal_ultrasound \
  --labels_path ./acouslic_dataset/labels.csv \
  --test_scans_path ./acouslic_dataset/cross_valid_folds/0/test \
  --relevance_model ./part1_frame_classification/output/relevance/relevance_model.pt \
  --quality_model   ./part1_frame_classification/output/quality/quality_model.pt \
  --output ./part1_frame_classification/output/hier/results/predictions.csv \
  --arch_rel convnext \
  --arch_qual densenet \
  --rel_threshold 0.0 \
  --sub_score 0.6 \
  --use_gpu true
```
---

#  Results (5-Fold Cross-Validation)

### 3-Class Classifier

| network       | Loss        | Mean WFSS   | Optimal Hit Rate   |
|---------------|-------------|-------------|--------------------|
| EfficientNet  | CE_WFSS     |    0.626    |        0.6056      |
| ConvNeXt      | CE_WFSS     |    0.6795   |        0.6592      |
| DenseNet121   | CE_WFSS     |    0.6754   |        0.6694      |
| ResNet50      | CE_WFSS     |    0.5965   |        0.5265      |

| network       | Loss        | Mean WFSS   | Optimal Hit Rate   |
|---------------|-------------|-------------|--------------------|
| ConvNeXt      | CE          |    0.6344   |        0.6056      |
| ConvNeXt      | focal       |    0.7562   |        0.8375      |
| ConvNeXt      | CE_WFSS     |    0.6667   |        0.8310      |
| ConvNeXt      | focal_WFSS  |    0.7112   |        0.8296      |

### Hierarchical Model

| Configuration                         | Mean WFSS   | Optimal Hit Rate   |
|---------------------------------------|-------------|--------------------|
| Relevance + Quality (convex+dense)    |  0.7737     |        0.7615      |  

- I havent tried the Hierarchical model with focal loss but i think it will get the same results or even better that the 3-class Classifier with focal loss
---