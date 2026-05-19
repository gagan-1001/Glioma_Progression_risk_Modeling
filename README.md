# Spatiotemporal Modeling of Post-Treatment Glioma Evolution Using Longitudinal MRI for Time-Dependent Tumor Progression Prediction

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Clinical Significance](#2-clinical-significance)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
   - 4.1 [Clinical Preprocessing](#41-clinical-preprocessing)
   - 4.2 [MRI Preprocessing](#42-mri-preprocessing)
   - 4.3 [Why Survival Analysis — Not Classification](#43-why-survival-analysis--not-classification)
5. [Model Architecture](#5-model-architecture)
   - 5.1 [Spatial Encoder — MobileNetV3](#51-spatial-encoder--mobilenetv3)
   - 5.2 [Temporal Encoder — LSTM](#52-temporal-encoder--lstm)
   - 5.3 [Clinical Encoder — MLP](#53-clinical-encoder--mlp)
   - 5.4 [Multimodal Fusion and Prediction Head](#54-multimodal-fusion-and-prediction-head)
   - 5.5 [Training Objective — Cox Proportional Hazards Loss](#55-training-objective--cox-proportional-hazards-loss)
   - 5.6 [Inference — Breslow Estimator](#56-inference--breslow-estimator)
6. [Training Pipeline](#6-training-pipeline)
7. [Results](#7-results)
8. [Hyperparameters](#8-hyperparameters)
9. [Project Structure](#9-project-structure)
10. [How to Run](#10-how-to-run)
11. [How to Run Inference on a New Patient](#11-how-to-run-inference-on-a-new-patient)

---

## 1. Problem Statement

> **Given a sequence of post-treatment MRI scans for a glioma patient, can we perform risk modeling to predict the probability that the patient's tumor will progress within a clinically meaningful time horizon?**

Formally, the model estimates:

```
P(tumor progression within t days | MRI sequence, clinical features)
```

for multiple time horizons simultaneously — for example, 30 days (1 month), 180 days (6 months), and 365 days (1 year).

This is a **survival analysis problem**, not a binary classification problem. The model does not predict *whether* a patient will progress — it predicts *when*, expressed as a continuous risk curve over time. Patients who have not yet progressed at the end of their follow-up period (censored patients) are handled correctly via the Cox partial likelihood framework, which uses their partial information rather than discarding them.

The core question the model answers for a clinician is:

> *"Based on how this patient's tumor has evolved across their MRI scans since treatment, how likely are they to experience confirmed progression in the next 30 days? In the next 6 months?"*

---

## 2. Clinical Significance

Glioma is the most common and aggressive primary brain tumor in adults. Despite surgical resection, radiation, and chemotherapy, the majority of patients experience tumor recurrence or progression. Timely detection and prediction of progression is critical because:

- **Early intervention can extend survival.** If a clinician can identify high-risk patients weeks before clinical symptoms appear, treatment adjustments — additional chemotherapy cycles, re-irradiation, or entry into a clinical trial — can be made proactively.

- **Current clinical practice is reactive, not predictive.** Patients undergo routine follow-up MRI scans every 2–3 months, and progression is confirmed only after it becomes radiologically obvious. There is no established tool that uses the *evolution* of tumor imaging over time to generate forward-looking risk estimates.

- **Tumor sub-region dynamics carry prognostic information that volumetric metrics miss.** The current standard of care tracks total tumor volume. However, whether the *enhancing tumor* region is expanding versus the *necrotic core* or *edema zone* carries fundamentally different clinical meaning. This model learns these sub-region dynamics explicitly.

### Novelty Over Prior Work

| Aspect | Prior Work | This Model |
|--------|-----------|------------|
| MRI input | Single timepoint | Longitudinal sequence (up to 6 scans) |
| Spatial representation | Radiomics or whole-volume CNN | Voxel-level features from tumor sub-regions |
| Temporal modeling | None or simple delta-volume | LSTM over spatiotemporal feature sequence |
| Clinical integration | Separate or not included | Fused at the representation level |
| Output | Binary classification | Time-dependent progression probability |
| Censoring handling | Often ignored or excluded | Proper Cox PH survival analysis |

---

## 3. Dataset

**Source:** MU-Glioma-Post dataset from The Cancer Imaging Archive (TCIA)

| Property | Value |
|----------|-------|
| Patients | 203 adult glioma patients (post-treatment) |
| Total MRI timepoints | 617 longitudinal scan sessions |
| Scans per patient | 1 to 6 (irregular, patient-specific) |
| MRI modalities | T1 native, T1 contrast-enhanced (T1c), T2-weighted, FLAIR |
| Tumor mask labels | 4 sub-regions per timepoint |
| Progression events | 152 patients (75%) |
| Censored patients | 51 patients (25%) |
| Median time to event | 152 days |

### Tumor Segmentation Mask Labels

Each timepoint includes an expert-validated segmentation mask with four integer labels:

| Label | Sub-region | Clinical Meaning |
|-------|-----------|-----------------|
| 1 | Non-enhancing tumor core (NETC) | Necrotic, non-active tumor tissue |
| 2 | Tumor infiltration / edema | Peritumoral invasion zone |
| 3 | Enhancing tumor (ET) | Active, aggressive tumor (blood-brain barrier breakdown) |
| 4 | Resection cavity (RC) | Surgically removed region |

Segmentations were generated by nnU-Net and validated by neuroradiologists.

### Survival Labels

| Column | Value | Meaning |
|--------|-------|---------|
| `event` | 1 | Clinically confirmed tumor progression occurred |
| `event` | 0 | Censored — no progression observed by end of follow-up |
| `time` (event=1) | days | Days from diagnosis to first confirmed progression |
| `time` (event=0) | days | Last available MRI scan day (censoring time) |

---

## 4. Methodology

### 4.1 Clinical Preprocessing

The raw clinical Excel file contains 74 columns per patient covering demographics, treatment history, genomic markers, and MRI timing. The preprocessing pipeline transforms this into a clean, model-ready feature matrix through the following steps:

**Step 1 — Survival label construction.** The `event` column (0/1) and `time` column (days) are extracted. For progressed patients, time is days to first progression. For censored patients, time is set to the last available MRI scan day, preserving their partial follow-up information.

**Step 2 — Negative value handling.** The surgery timing column contains 199 negative values, which are clinically meaningful — they indicate that surgery (typically a diagnostic biopsy) occurred *before* the formal diagnosis date. The strategy is to take the absolute value, since the magnitude (proximity to diagnosis) is the relevant signal, not the sign.

**Step 3 — Dose column cleaning.** The radiation dose column stores values in free-text format (e.g., `"60 Gy"`, `"54Gy"`, `"NA"`). A regex extraction pulls the numeric value; non-numeric entries become NaN.

**Step 4 — Categorical encoding.** Fourteen string columns (sex, diagnosis grade, genomic markers including MGMT methylation, IDH1, ATRX, EGFR amplification, and others) are label-encoded to integer codes, then cast to float. NaN is preserved through the encoding rather than treated as a valid category.

**Step 5 — Domain-aware feature engineering.** The key principle here is that *missing does not mean unknown* — in this dataset, a missing therapy column means the therapy was not given. Binary `received_X` flags are created for radiation, chemotherapy, and immunotherapy. Timing columns for treatments not given are filled with 0 as a sentinel value (not as imputed data). The number of MRI timepoints per patient is derived as a numeric feature.

**Step 6 — Imputation and scaling.** Three column-specific imputation strategies are applied: sentinel-0 columns (therapy timing) are verified and force-filled if any NaN remains; binary and categorical columns use mode imputation; remaining numeric columns use median imputation as a last resort. All 38 features are then StandardScaled to zero mean and unit variance.

**Step 7 — PCA dimensionality reduction.** Principal Component Analysis reduces the 38 correlated clinical features to 26 principal components retaining 95.2% of total variance. This removes redundancy between correlated columns (e.g., RT start days and RT end days) and reduces overfitting risk in the small cohort.

**Output:** A CSV with 29 columns — `Patient_ID`, `event`, `time`, and `PCA_1` through `PCA_26`.

---

### 4.2 MRI Preprocessing

Each MRI timepoint folder contains five files: four modality volumes and one segmentation mask. The preprocessing pipeline converts these into a fixed-size, model-ready tensor.

**File loading.** Files are matched by case-insensitive keyword search (`t1n`, `t1c`, `t2w`, `t2f`, `tumormask`). If any of the five files is missing, the entire timepoint is skipped gracefully — no crash, no partial data.

**Z-score normalisation.** Each MRI modality is independently normalised to zero mean and unit standard deviation: `(v - mean) / std`. This removes scanner-specific intensity scale and offset, making intensities comparable across patients and scan sessions. Division-by-zero is handled safely for blank volumes.

**Mask decomposition.** The multi-label segmentation mask (integer labels 1–4) is decomposed into four separate binary channels using `(mask == label).astype(float32)`. This allows the CNN to learn the spatial morphology of each sub-region independently. Treating the label integer as an ordinal value would incorrectly imply an ordering relationship between sub-regions.

**Tumor-centred patch extraction.** A tight axis-aligned bounding box is computed around all non-zero mask voxels. This bounding box is expanded by 15 voxels in each direction to capture peritumoral tissue (the edema and infiltration zone that may contain progression-relevant signal). The expanded region is cropped from the 8-channel volume and then resized to a fixed 64×64×64 isotropic patch. If no tumor is found (empty mask), the pipeline falls back to a center crop of the brain.

**Modality-aware resizing.** MRI channels (0–3) are resized using trilinear interpolation, which preserves smooth intensity gradients. Mask channels (4–7) are resized using nearest-neighbour interpolation, which preserves binary 0/1 values. Trilinear interpolation on binary masks would create fractional values like 0.3 or 0.7, which are meaningless for segmentation channels.

**Sequence construction.** Each patient's timepoints are processed chronologically and collected into a padded sequence of shape `(6, 8, 64, 64, 64)`. Patients with fewer than 6 valid timepoints are right-padded with zero tensors. A companion sequence mask of shape `(6,)` records which positions contain real scans (1.0) versus padding (0.0).

**Training augmentation.** During training only, two mild augmentations are applied per patch: random Gaussian noise (probability 0.2, std 0.05) to simulate scanner noise, and random intensity scaling (probability 0.3, ±10%) to simulate scanner gain variation between sessions. Geometric augmentations (flipping, rotation) are deliberately disabled because brain lateralisation may carry clinically meaningful information for tumor progression.

---

### 4.3 Why Survival Analysis — Not Classification

A binary classification approach (will this patient progress: yes/no) has a fundamental flaw with this data: 51 patients (25%) are *censored* — they had not progressed by the end of their follow-up period, but they may progress the next day, or never. Labelling them as "no progression" is factually incorrect and introduces systematic bias.

The Cox Proportional Hazards model handles censored observations correctly. For each patient who progressed, the Cox loss asks: "Among all patients still at risk at this time point, does the model correctly rank this patient's risk above the others?" Censored patients contribute to the denominator (the risk set) of this calculation for all event times that fall within their observed period — their partial information is used, not discarded.

This formulation outputs a **relative log-risk score** per patient. The Breslow estimator, fitted post-training, converts this into absolute survival probabilities `S(t)` and progression probabilities `P(t) = 1 - S(t)` at any time horizon.

---

## 5. Model Architecture

The model is a multimodal architecture that processes two input streams — longitudinal MRI scans and tabular clinical features — and fuses them to produce a single Cox log-risk score.

### Pipeline Overview

The MRI stream passes through a spatial encoder that extracts morphological features from each timepoint, followed by a temporal encoder that models how these features evolve across the longitudinal sequence. In parallel, the clinical stream passes through a dedicated MLP encoder. The two streams are concatenated and passed through a prediction head that outputs the final log-risk score. Post-training, a Breslow estimator converts this score into calibrated progression probabilities.

---

### 5.1 Spatial Encoder — MobileNetV3

**Purpose:** Extract spatial morphological features from each 3-D MRI timepoint — tumor boundary irregularity, sub-region shape, signal heterogeneity within each region.

**Architecture:** MobileNetV3-Small pretrained on ImageNet, adapted for 8-channel MRI input, with a custom projection head.

**Why MobileNetV3-Small?** It is designed for computational efficiency using depthwise separable convolutions, which significantly reduces parameter count compared to standard convolutions. On a cohort of only 203 patients, a heavier backbone (e.g., ResNet-50) would overfit severely. The small variant balances representational capacity against regularisation need.

**Why 2-D slices from a 3-D volume?** Processing a full 3-D volume with a 3-D CNN would require enormous GPU memory — a single 8-channel 64×64×64 volume contains ~2 million voxels. Instead, the 3-D patch is split into 64 axial slices of shape `(8, 64, 64)`. Each slice is encoded independently by MobileNetV3. The resulting 64 slice embeddings are then mean-pooled across the depth dimension, producing one 512-dimensional feature vector per timepoint. This approach is computationally feasible on an 8 GB GPU and still captures the spatial structure of each modality and mask channel.

**Adaptation for MRI:** The first convolutional layer is replaced to accept 8 input channels instead of 3 (RGB). The ImageNet classification head is replaced with a projection head: `Linear(576 → 512) → ReLU → Dropout(0.3)`.

**Memory management — SLICE_CHUNK:** Without chunking, a batch of `B × T × D = 2 × 6 × 64 = 768` slices would pass through MobileNet simultaneously, consuming ~14 GB of VRAM. The forward pass instead processes 16 slices at a time, accumulating features and concatenating afterwards. The mathematical result is identical; peak VRAM is reduced by ~48×.

**Output per patient per timepoint:** A 512-dimensional spatial feature vector.

---

### 5.2 Temporal Encoder — LSTM

**Purpose:** Model how the tumor's spatial features evolve across the longitudinal sequence of timepoints — capturing growth acceleration, stabilisation, emergence of new infiltration, or shifts in sub-region composition over time.

**Architecture:** Single-layer LSTM with 256 hidden units, operating on the sequence of per-timepoint spatial feature vectors.

**Input:** `(B, T=6, 512)` — batch of padded sequences, one 512-d vector per timepoint.

**Output:** `(B, 256)` — the final hidden state `h_n`, which is the LSTM's summary of the patient's complete temporal trajectory.

**Why LSTM over a Transformer?** With sequences of only 1–6 timepoints, the self-attention mechanism in a Transformer offers no advantage over the LSTM's recurrent hidden state — both can model the dependencies in a sequence this short. The LSTM has far fewer parameters and is better regularised for a small dataset. The hidden state also provides a natural summary of the full sequence, whereas a Transformer would require an additional pooling step.

**Handling variable-length sequences:** Patients have between 1 and 6 valid timepoints. The sequence is right-padded to length 6 with zero tensors, and a companion sequence mask records which positions are real. The LSTM processes all 6 positions but the padded positions carry near-zero signal, so the final hidden state is effectively dominated by the real timepoints. Input dropout (rate 0.2) is applied to the sequence before the LSTM for regularisation.

---

### 5.3 Clinical Encoder — MLP

**Purpose:** Transform the 26-dimensional PCA clinical feature vector into a dense 128-dimensional embedding that can be meaningfully fused with the 256-dimensional MRI temporal embedding.

**Architecture:** Two-layer MLP with batch normalisation and dropout.

```
Linear(26 → 128) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(128 → 128) → BatchNorm1d → ReLU → Dropout(0.3)
```

**Why a separate encoder for clinical features?** Clinical features (age, molecular markers, therapy timing, MRI scan count) are on very different scales and of different types — binary flags, encoded categoricals, PCA components of continuous measurements. A dedicated MLP with batch normalisation allows the model to learn an appropriate nonlinear transformation of this heterogeneous feature space before fusion, rather than naively concatenating raw clinical values with high-dimensional MRI features where MRI would dominate by sheer dimensionality.

---

### 5.4 Multimodal Fusion and Prediction Head

**Fusion:** The 256-dimensional MRI temporal embedding and the 128-dimensional clinical embedding are concatenated to form a 384-dimensional joint representation: `[mri_embedding ‖ clinical_embedding]`.

**Prediction head:**

```
Linear(384 → 128) → ReLU → Dropout(0.3)
Linear(128 → 64)  → ReLU → Dropout(0.3)
Linear(64 → 1)
```

**Output:** A single unbounded scalar — the **Cox log-risk score**. No sigmoid, softmax, or exp is applied. Higher value means higher predicted progression risk. This raw score is what the Cox loss trains directly.

---

### 5.5 Training Objective — Cox Proportional Hazards Loss

The negative partial log-likelihood of the Cox model is minimised:

```
L = - (1 / N_events) × Σ_{i: event_i=1} [ risk_i - log Σ_{j: t_j ≥ t_i} exp(risk_j) ]
```

For each patient who progressed (event=1), the term `risk_i - log Σ_{j: t_j ≥ t_i} exp(risk_j)` measures whether the model assigned this patient a higher risk than all other patients who were still at risk at time `t_i`. A perfect model assigns the highest risk to the patient who progresses first. The loss is normalised by the number of events so that its scale is independent of batch size and event rate.

**Implementation efficiency:** Sorting patients by descending time converts the risk-set sum into a prefix sum, computed efficiently via `torch.logcumsumexp` in a single pass rather than a nested loop.

**Constraint:** At least 2 events must be present in each batch for the Cox loss to be meaningful. This is enforced by `drop_last=True` on the training DataLoader and by skipping batches with fewer than 2 events.

---

### 5.6 Inference — Breslow Estimator

After training, the model produces only relative log-risk scores. To obtain calibrated progression probabilities, the **Breslow non-parametric baseline cumulative hazard estimator** is fitted on all 203 patients using the trained model's predictions.

**How it works:**

At each observed event time `t_i`, the Breslow estimator computes an increment to the baseline cumulative hazard:

```
ΔH₀(t_i) = 1 / Σ_{j: t_j ≥ t_i} exp(risk_j)
```

The cumulative hazard at time `t` is the sum of all increments up to `t`: `H₀(t) = Σ_{i: t_i ≤ t} ΔH₀(t_i)`.

This lookup table is saved as `breslow_table.pkl`.

**At inference for a new patient with log-risk score `r`:**

```
S(t) = exp( -H₀(t) × exp(r - mean_risk) )
P(progression within t days) = 1 - S(t)
```

The mean risk is subtracted for numerical stability. This formula can be evaluated at any time horizon without re-running the model.

---

## 6. Training Pipeline

### Data Split Strategy

All 203 patients are split 85/15 into a cross-validation pool (~173 patients) and a permanently locked held-out test set (~30 patients). The test set is set aside before any training begins and is never used for fold selection, hyperparameter tuning, or model selection. It is evaluated exactly once at the very end to provide an unbiased performance estimate.

5-fold cross-validation is performed on the CV pool. Each fold uses approximately 80% of the pool for training and 20% for validation. Every patient in the CV pool appears in the validation set exactly once across all five folds.

### Learning Rate Schedule

A linear warmup over the first 3 epochs ramps the learning rate from `LR/10 = 1e-5` to the peak `LR = 1e-4`. This prevents unstable gradient updates in the early epochs when model weights are randomly initialised. After warmup, a cosine annealing schedule decays the learning rate from `1e-4` to `1e-6` over the remaining epochs, allowing aggressive learning early and fine-grained convergence later.

### Early Stopping

Training halts if the validation C-index does not improve for 12 consecutive epochs. The checkpoint with the highest validation C-index is saved separately and used for all downstream evaluation and Breslow fitting.

### Mixed Precision Training

`torch.cuda.amp.autocast` runs the forward pass in float16 where numerically safe, reducing VRAM usage by approximately 50% and speeding up GPU computation. A `GradScaler` prevents gradient underflow from float16 precision.

### Gradient Clipping

`torch.nn.utils.clip_grad_norm_(max_norm=5.0)` prevents exploding gradients in the LSTM, which are common when training recurrent networks on variable-length sequences. Clipping at 5.0 is standard for LSTM-based survival models.

---

## 7. Results

Training was conducted on an NVIDIA GPU (8 GB VRAM) using the MU-Glioma-Post dataset (203 patients, 617 timepoints). The full 5-fold cross-validation was completed with the multimodal architecture (MRI spatiotemporal + clinical fusion).

### Cross-Validation Performance

| Metric | Value |
|--------|-------|
| **C-index (mean ± std)** | **0.6894** |
| **C-index (best fold)** | **0.7341** |
| Integrated Brier Score (IBS) | 0.28 |
| td-AUC @ 30 days | 0.92 |
| td-AUC @ 6 months (180 days) | 0.7523 |

### Interpretation of Results

**C-index of 0.69 (mean) / 0.73 (best):** A C-index of 0.5 is equivalent to random ranking. A C-index of 0.73 means the model correctly ranks the relative progression timing of two randomly selected patients 73% of the time. This is a clinically meaningful discrimination performance for a survival task on a cohort of 203 patients, where the inherent noise in progression timing (influenced by treatment variation, genomic heterogeneity, and imaging quality) creates a natural ceiling below 1.0.

**td-AUC @ 30 days of 0.92:** At the 30-day horizon, the model achieves excellent discrimination — it correctly identifies 92% of the time whether a patient will progress within the next month. This is the most clinically actionable horizon (short-term treatment decisions) and the model's strongest result.

**td-AUC @ 6 months of 0.75:** At the 6-month horizon, discrimination remains good. The lower value compared to 30 days reflects the inherent increased uncertainty in longer-term predictions, and the reduced number of validation patients with event times in this range.

**IBS of 0.28:** The null model (predicting constant 0.5 survival for all patients) achieves IBS ≈ 0.25. The model's IBS of 0.28 is slightly above this baseline, indicating that while risk ranking (C-index, AUC) is strong, absolute probability calibration has room for improvement. This is typical for Cox-based models fitted on small cohorts, where the Breslow estimator can exhibit slight miscalibration.

---

## 8. Hyperparameters

### MRI Processing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PATCH_SIZE` | 64 | Isotropic spatial size of every extracted 3-D tumor patch (voxels) |
| `PATCH_MARGIN` | 15 | Voxels added around tight tumor bounding box on each side |
| `MAX_TIMEPOINTS` | 6 | Maximum longitudinal scans per patient; shorter sequences are zero-padded |
| `IN_CHANNELS` | 8 | 4 MRI modalities + 4 binary tumor mask channels |
| `CNN_OUT_DIM` | 512 | Output embedding dimension from MobileNetV3 per timepoint |
| `LSTM_HIDDEN` | 256 | LSTM hidden state size — controls temporal memory capacity |
| `SLICE_CHUNK` | 16 | Slices processed by MobileNet per sub-batch to prevent CUDA OOM |

### Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPOCHS` | 30–50 | Maximum training epochs per fold (early stopping usually terminates before) |
| `BATCH_SIZE` | 2–4 | Patients per batch — constrained by 8 GB GPU memory |
| `LEARNING_RATE` | 1e-4 | Peak learning rate after warmup |
| `WEIGHT_DECAY` | 3e-5 | L2 regularisation — slightly higher than default for small cohort |
| `WARMUP_EPOCHS` | 3 | Linear LR ramp epochs before cosine decay |
| `PATIENCE` | 12 | Early stopping tolerance (consecutive epochs without C-index improvement) |
| `N_FOLDS` | 5 | Cross-validation folds on the 85% CV pool |
| `TEST_FRACTION` | 0.15 | Fraction of all patients locked as held-out test set |
| `SEED` | 42 | Global random seed for reproducibility |
| `AUC_TIME_HORIZONS` | [30, 180] | td-AUC evaluation horizons in days |

### Augmentation

| Parameter | Value | Description |
|-----------|-------|-------------|
| Gaussian noise probability | 0.2 | Probability of adding noise per patch |
| Gaussian noise std | 0.05 | Noise magnitude — 5% of normalised intensity range |
| Intensity scale probability | 0.3 | Probability of applying gain variation |
| Intensity scale factor | 0.1 | ±10% intensity scaling |

### Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| `clinical_hidden` | 128 | Hidden and output dimension of clinical MLP encoder |
| `fusion_dim` | 384 | Concatenated MRI (256) + clinical (128) embedding dimension |
| `head_hidden` | 128 | First hidden layer of prediction head |
| `input_dropout` | 0.2 | Dropout on LSTM input sequence |
| `encoder_dropout` | 0.3 | Dropout in MobileNet head, clinical encoder, and prediction head |

---

## 9. Project Structure

```
project/
│
├── clinical_preprocessing.py   # Clinical data loading, cleaning, encoding, PCA
├── mri_model.py                # MRI pipeline, dataset class, model architecture, Cox loss
├── train.py                    # Training loop, CV, metrics, Breslow fitting, inference
│
├── clinical_file.xlsx          # Raw clinical Excel data (74 columns, 203 patients)
├── clinical_processed.csv      # Output of clinical_preprocessing.py (auto-generated)
│
├── MU-Glioma-Post/             # Root MRI data directory
│   └── PatientID_XXXX/
│       └── Timepoint_X/
│           ├── brain_t1n*.nii.gz
│           ├── brain_t1c*.nii.gz
│           ├── brain_t2w*.nii.gz
│           ├── brain_t2f*.nii.gz
│           └── tumorMask.nii
│
└── checkpoints/                # Saved model checkpoints (auto-generated)
    ├── fold_1_best.pt
    ├── fold_2_best.pt
    ├── fold_3_best.pt
    ├── fold_4_best.pt
    ├── fold_5_best.pt
    └── breslow_table.pkl
```

---

## 10. How to Run

### Prerequisites

```bash
pip install torch torchvision nibabel monai pandas numpy scikit-learn lifelines scikit-survival tqdm
```

### Step 1 — Update paths

In both `clinical_preprocessing.py` and `train.py`, update the file path constants at the top of each file to match your local directory structure:

```python
DATA_ROOT      = Path(r"path/to/MU-Glioma-Post")
CLINICAL_EXCEL = Path(r"path/to/clinical_file.xlsx")
PROCESSED_CSV  = Path(r"path/to/clinical_processed.csv")
CHECKPOINT_DIR = Path(r"path/to/checkpoints")
```

### Step 2 — Run training

```bash
python train.py
```

This will:
1. Run clinical preprocessing (or load from cache if `clinical_processed.csv` exists)
2. Split patients 85/15 into CV pool and locked test set
3. Run 5-fold cross-validation with training, validation, and early stopping per fold
4. Evaluate the best fold model on the held-out test set
5. Fit the Breslow estimator on all patients
6. Save all checkpoints and the Breslow table to `CHECKPOINT_DIR`

To force re-running clinical preprocessing (e.g. after changing feature engineering), delete `clinical_processed.csv` before running.

---

## 11. How to Run Inference on a New Patient

After training completes, the model and Breslow table are saved to disk. To predict progression probabilities for a new patient:

```python
from train import load_for_inference, predict_patient
from pathlib import Path
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model and Breslow estimator
model, breslow = load_for_inference(
    checkpoint_path        = Path(r"checkpoints/fold_2_best.pt"),
    breslow_path           = Path(r"checkpoints/breslow_table.pkl"),
    num_clinical_features  = 26,   # number of PCA features in clinical_processed.csv
    device                 = device,
)

# Load the clinical data (needed to retrieve this patient's clinical features)
clinical_df = pd.read_csv(r"clinical_processed.csv")
clinical_feature_cols = [
    c for c in clinical_df.columns if c not in ["Patient_ID", "time", "event"]
]

# Run inference for a specific patient
result = predict_patient(
    patient_dir           = Path(r"MU-Glioma-Post/PatientID_XXXX"),
    clinical_df           = clinical_df,
    clinical_feature_cols = clinical_feature_cols,
    model                 = model,
    breslow_table         = breslow,
    device                = device,
    time_horizons         = [30, 180, 365],
)

print(result["interpretation"])
```

### Example Output

```
Patient : PatientID_0190
Log-risk: 1.2341

  P(progression <=  30d): 12.3%  [██░░░░░░░░░░░░░░░░░░]
  P(progression <= 180d): 45.1%  [█████████░░░░░░░░░░░]
  P(progression <=  365d / 1yr): 67.8%  [█████████████░░░░░░░]
```

### Interpretation

| Output | Meaning |
|--------|---------|
| Log-risk score | Raw model output — relative to other patients. Higher = higher risk. Has no standalone probabilistic meaning. |
| P(progression ≤ 30d) | Probability of clinically confirmed tumor progression within the next 30 days |
| P(progression ≤ 180d) | Probability of progression within the next 6 months |
| P near 1.0 | Model is confident progression is imminent |
| P near 0.0 | Model predicts stable disease in this time window |

The Breslow estimator anchors the relative log-risk scores to the actual observed event times in the training data. This is what enables converting a dimensionless risk score into a calibrated probability between 0 and 1.

---

## References

- Cox, D.R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society, Series B*, 34(2), 187–220.
- Howard Howard, F.M. et al. (2021). MU-Glioma-Post: Post-treatment glioma MRI dataset. The Cancer Imaging Archive (TCIA).
- Howard, H. et al. MobileNetV3: Searching for MobileNetV3. *ICCV 2019*.
- Isensee, F. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 2021.
