"""
train.py
========
Main training + validation script for the glioma spatiotemporal survival model.

Pipeline:
    1. Load (or build) preprocessed clinical CSV
    2. Scan MRI root directory for all patient folders
    3. Lock away 15% of patients as a held-out test set (never touched during training)
    4. Run 5-fold cross-validation on the remaining 85% of patients
    5. Per fold: train with Cox PH loss, validate with C-index + td-AUC + IBS
    6. After all folds: evaluate on locked test set using best fold model
    7. Fit Breslow estimator on full dataset → save for inference

Usage:
    python train.py

──────────────────────────────────────────────────────────────────────────────
HOW THE MODEL OUTPUT BECOMES A PROBABILITY
──────────────────────────────────────────────────────────────────────────────
The model outputs a raw log-risk score (single scalar per patient).
This is RELATIVE — higher = higher risk — but has no probabilistic meaning alone.

To get P(progression within X days) we use the Breslow estimator:

    Step 1  Train Cox model (done below).
    Step 2  Fit Breslow baseline cumulative hazard H0(t) on the full dataset.
    Step 3  For a new patient with log-risk score r:
                S(t)  = exp( -H0(t) x exp(r) )        <- survival probability
                P(t)  = 1 - S(t)                       <- progression probability
    Step 4  Query at any horizon:
                P(progression within 30 days)  = 1 - S(30)
                P(progression within 180 days) = 1 - S(180)

Breslow table is saved to disk after training. At inference, load model
checkpoint + breslow table -> call predict_patient().
──────────────────────────────────────────────────────────────────────────────
"""

import os
import math
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from tqdm import tqdm
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from sklearn.model_selection import KFold

# ── Local modules ──────────────────────────────────────────────────────────
from clinical_preprocessing import preprocess_clinical, OUTPUT_CSV, CLINICAL_FILE
from mri_model import (
    GliomaDataset,
    GliomaModel,
    cox_ph_loss,
    build_patient_index,
    build_patient_sequence,
    MAX_TIMEPOINTS,
)


# ===========================================================================
# CONFIGURATION — edit only this section
# ===========================================================================

# Path to the root MRI data directory (contains one subfolder per patient)
DATA_ROOT = Path(r"C:\Users\sanat\Downloads\MU-Glioma-Post")

# Path to the raw clinical Excel file (input to clinical_preprocessing.py)
CLINICAL_EXCEL = Path(r"C:\Users\sanat\Desktop\MLPR_Mobile_Small\clinical_file.xlsx")

# Path where the preprocessed clinical CSV will be saved / loaded from
PROCESSED_CSV  = Path(r"C:\Users\sanat\Desktop\MLPR_Mobile_Small\clinical_processed.csv")

# Directory where model checkpoints will be saved (one .pt file per fold)
CHECKPOINT_DIR = Path(r"C:\Users\sanat\Desktop\MLPR_Mobile_Small\checkpoints")

# ── Training hyperparameters ───────────────────────────────────────────────
EPOCHS        = 30       # Max epochs per fold. Early stopping (PATIENCE=12) will
                          # usually terminate before this on a well-converged fold.

BATCH_SIZE    = 4        # Patients per batch. Keep at 2 for 8GB GPU.
                          # SLICE_CHUNK in mri_model.py handles the actual VRAM cost.

LEARNING_RATE = 1e-4     # Peak LR reached after warmup. AdamW optimizer.

WEIGHT_DECAY  = 3e-5     # Slightly higher than 1e-5 for better generalisation
                          # on this small cohort (203 patients total).

WARMUP_EPOCHS = 3        # LR ramps linearly from LR/10 -> LR over these epochs,
                          # then cosine decays. Prevents early gradient instability.

PATIENCE      = 12       # Early stopping: halt fold if val C-index doesn't improve
                          # for this many consecutive epochs.

N_FOLDS       = 3       # Cross-validation folds on the 85% CV pool.
                          # Each fold: ~80% train, ~20% val within the pool.

TEST_FRACTION = 0.15     # Fraction of all patients locked away as held-out test set.
                          # Never seen during training or fold selection (~30 patients).

SEED          = 42       # Controls all random splits, shuffles, and weight init.

NUM_WORKERS   = 0        # DataLoader workers. Keep 0 on Windows to avoid issues.

# ── Evaluation time horizons ───────────────────────────────────────────────
# td-AUC evaluated at these days. The function automatically skips any horizon
# outside the observed event range in a given fold — eliminates nan warnings.
AUC_TIME_HORIZONS = [30, 180]   # 30 days (1 month) and 180 days (6 months)


# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# LR SCHEDULER: LINEAR WARMUP + COSINE DECAY
# ===========================================================================

def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-6):
    """
    LambdaLR scheduler with linear warmup followed by cosine annealing.

    Epochs 1 ... warmup_epochs : LR ramps from LR/10 up to LR linearly.
    Epochs warmup+1 ... total  : LR follows cosine curve down to eta_min.

    Why warmup?
        In the first few epochs the model has random weights and gradients
        are noisy. A large LR at this stage can push weights into bad basins.
        Warmup lets the optimizer take small, stable steps initially.

    Args:
        optimizer     : AdamW optimizer instance
        warmup_epochs : Number of warmup epochs (WARMUP_EPOCHS = 3)
        total_epochs  : Total planned training epochs (EPOCHS = 50)
        eta_min       : Minimum LR at end of cosine decay

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    def lr_lambda(epoch):
        # LambdaLR passes 0-indexed epoch
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / max(warmup_epochs - 1, 1)
        else:
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_frac = eta_min / LEARNING_RATE
            return min_frac + (1.0 - min_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# DATA HELPERS
# ===========================================================================

def load_or_build_clinical(
    clinical_excel: Path = CLINICAL_EXCEL,
    processed_csv:  Path = PROCESSED_CSV,
) -> pd.DataFrame:
    """
    Return preprocessed clinical DataFrame from cache or by running pipeline.
    Delete PROCESSED_CSV to force re-preprocessing from raw Excel.
    """
    if processed_csv.exists():
        print(f"[INFO] Found existing clinical CSV — loading: {processed_csv}")
        return pd.read_csv(processed_csv)
    print("[INFO] Preprocessed CSV not found — running clinical_preprocessing ...")
    return preprocess_clinical(filepath=clinical_excel, output_csv=processed_csv)


def split_patients_85_15(
    all_patients:  list,
    test_fraction: float = TEST_FRACTION,
    seed:          int   = SEED,
) -> tuple:
    """
    Split all patients into a CV pool (85%) and a locked held-out test set (15%).

    The test set is set aside immediately and evaluated exactly once at the
    very end of training. This gives an unbiased performance estimate that
    is not contaminated by any fold selection decisions.

    Why 85/15 instead of all-patients CV?
        Evaluating on patients whose fold appeared in training inflates
        metrics. A locked test set gives a clean final number.

    Args:
        all_patients  : All patient directory name strings (sorted)
        test_fraction : Fraction to lock away (default 0.15 = ~30 patients)
        seed          : Random seed for reproducibility

    Returns:
        (cv_ids, test_ids) — both as lists of patient directory name strings
    """
    ids = list(all_patients)
    random.seed(seed)
    random.shuffle(ids)
    n_test   = int(len(ids) * test_fraction)
    test_ids = ids[:n_test]
    cv_ids   = ids[n_test:]
    print(f"[INFO] Total patients : {len(ids)}")
    print(f"[INFO] CV pool        : {len(cv_ids)}  (used for {N_FOLDS}-fold CV)")
    print(f"[INFO] Test set       : {len(test_ids)}  (locked, evaluated once at end)\n")
    return cv_ids, test_ids


# ===========================================================================
# METRICS — nan-safe td-AUC and IBS
# ===========================================================================

def compute_tdauc(
    train_times:   np.ndarray,
    train_events:  np.ndarray,
    val_times:     np.ndarray,
    val_events:    np.ndarray,
    val_risks:     np.ndarray,
    time_horizons: list = AUC_TIME_HORIZONS,
) -> dict:
    """
    Compute time-dependent AUC at specified horizons with robust filtering.

    ROOT CAUSE OF THE nan WARNINGS IN THE PREVIOUS CODE:
        sksurv.cumulative_dynamic_auc() raises an exception if any requested
        horizon >= the largest observed EVENT time in the validation set.
        Fix: filter horizons to be strictly inside (t_min_val, t_max_event_val).
        Horizons outside this range are silently skipped — no nans, no warnings.

    Args:
        train_times   : (N_train,) observed times — needed for IPCW correction
        train_events  : (N_train,) event flags 0/1
        val_times     : (N_val,)   observed times
        val_events    : (N_val,)   event flags
        val_risks     : (N_val,)   model log-risk scores
        time_horizons : Candidate days (default: AUC_TIME_HORIZONS = [30, 180])

    Returns:
        dict {horizon_days: auc_value} for computable horizons only.
        Empty dict if no valid horizons or computation fails.
    """
    if val_events.sum() == 0:
        return {}

    train_surv = Surv.from_arrays(train_events.astype(bool), train_times)
    val_surv   = Surv.from_arrays(val_events.astype(bool),   val_times)

    # sksurv requires: t_min_val < horizon < t_max_event_val  (strict)
    t_min_val       = val_times.min()
    t_max_event_val = val_times[val_events == 1].max()

    valid_horizons = [t for t in time_horizons if t_min_val < t < t_max_event_val]

    if not valid_horizons:
        return {}   # No valid horizons — skip silently, no warning printed

    try:
        auc, _ = cumulative_dynamic_auc(train_surv, val_surv, val_risks, valid_horizons)
        return {t: float(a) for t, a in zip(valid_horizons, auc)}
    except Exception:
        return {}


def compute_ibs(
    train_times:  np.ndarray,
    train_events: np.ndarray,
    val_times:    np.ndarray,
    val_events:   np.ndarray,
    val_risks:    np.ndarray,
) -> float:
    """
    Integrated Brier Score — measures calibration quality.

    Lower = better. Null model (constant prediction) gives IBS ~0.25.
    A model with IBS < 0.25 is better-calibrated than random.

    ROOT CAUSE OF THE nan WARNINGS IN THE PREVIOUS CODE:
        integrated_brier_score() fails when integration range extends beyond
        the largest observed event time. Fix: clamp t_max to t_max_event - 1.

    Args:
        train_times, train_events : Training set survival data (for IPCW)
        val_times, val_events     : Validation set survival data
        val_risks                 : Model log-risk scores for val patients

    Returns:
        Float IBS value, or float("nan") if not computable.
    """
    if val_events.sum() == 0:
        return float("nan")

    train_surv = Surv.from_arrays(train_events.astype(bool), train_times)
    val_surv   = Surv.from_arrays(val_events.astype(bool),   val_times)

    # Clamp safely inside observed event time range
    t_min = val_times.min() + 1
    t_max = val_times[val_events == 1].max() - 1

    if t_min >= t_max:
        return float("nan")

    times = np.linspace(t_min, t_max, 50)

    try:
        # Approximate S(t) for IBS monitoring (not used at inference)
        base_surv = np.exp(-np.exp(val_risks - val_risks.mean()))
        survs     = np.stack([base_surv ** (t / t_max) for t in times], axis=1)
        survs     = np.clip(survs, 1e-6, 1 - 1e-6)
        return float(integrated_brier_score(train_surv, val_surv, survs, times))
    except Exception:
        return float("nan")


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def train_one_epoch(
    model:     torch.nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """
    Execute one full training epoch.

    Per batch:
        1. Forward pass through GliomaModel -> log-risk scores
        2. Cox PH loss (skip if < 2 events in batch)
        3. Mixed-precision backward pass with gradient clipping (max_norm=5.0)
        4. AdamW parameter update

    Returns:
        Mean Cox loss per batch over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for x, seq_mask,clinical_feat, time, event in tqdm(loader, desc="  Training", leave=False):
        x        = x.to(device, non_blocking=True)
        seq_mask = seq_mask.to(device, non_blocking=True)
        clinical_feat = clinical_feat.to(device, non_blocking = True)
        time     = time.to(device, non_blocking=True)
        event    = event.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            risk = model(x, seq_mask, clinical_feat).squeeze(1)
            if event.sum() < 2:
                continue
            loss = cox_ph_loss(risk, time, event)

        if loss.item() == 0.0:
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ===========================================================================
# VALIDATION LOOP
# ===========================================================================

@torch.no_grad()
def validate(
    model:        torch.nn.Module,
    loader:       DataLoader,
    device:       torch.device,
    train_times:  np.ndarray = None,
    train_events: np.ndarray = None,
) -> dict:
    """
    Evaluate model on a DataLoader. Returns C-index, td-AUC, and IBS.

    Metrics:
        c_index : Probability of correctly ranking two random patients.
                  0.5 = random, 1.0 = perfect, higher is better.
        td_auc  : dict {days: auc}. Only horizons within observed event
                  range are included — no nans, no warnings.
        ibs     : Integrated Brier Score. Lower is better (~0.25 = null model).

    Args:
        model, loader, device : Standard eval setup
        train_times, train_events : Required for td-AUC/IBS IPCW correction

    Returns:
        dict with keys "c_index", "td_auc", "ibs"
    """
    model.eval()
    all_risk, all_time, all_event = [], [], []

    for x, seq_mask,clinical_feat, time, event in loader:
        x    = x.to(device)
        seq_mask=seq_mask.to(device)
        clinical_feat=clinical_feat.to(device)
        risk = model(x, seq_mask,clinical_feat).squeeze(1)
        all_risk.extend(risk.cpu().numpy().tolist())
        all_time.extend(time.numpy().tolist())
        all_event.extend(event.numpy().tolist())

    risks  = np.array(all_risk)
    times  = np.array(all_time)
    events = np.array(all_event)

    try:
        # Negate risk: lifelines concordance_index expects higher=better outcome
        c_idx = concordance_index(times, -risks, events)
    except Exception:
        c_idx = float("nan")

    metrics = {"c_index": c_idx}

    if train_times is not None and train_events is not None:
        metrics["td_auc"] = compute_tdauc(train_times, train_events, times, events, risks)
        metrics["ibs"]    = compute_ibs(train_times, train_events, times, events, risks)

    return metrics


# ===========================================================================
# CHECKPOINT
# ===========================================================================

def save_checkpoint(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    c_index:   float,
    path:      Path,
) -> None:
    """Save model + optimizer state, epoch, and C-index to a .pt checkpoint file."""
    os.makedirs(path.parent, exist_ok=True)
    torch.save(
        {"epoch": epoch, "c_index": c_index,
         "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        path,
    )
    print(f"  ✔ Checkpoint saved -> {path}  (C-index: {c_index:.4f})")


# ===========================================================================
# SINGLE FOLD TRAINING
# ===========================================================================

def train_fold(
    fold:         int,
    train_ids:    list,
    val_ids:      list,
    clinical_df:  pd.DataFrame,
    device:       torch.device,
) -> dict:
    """
    Train and evaluate the model for one cross-validation fold.

    Steps:
        1. Build MRI patient index for train/val sets
        2. GliomaDataset: train with augmentation, val without
        3. DataLoaders: drop_last=True on train (avoids zero-event batches)
        4. Fresh model + AdamW + warmup/cosine scheduler
        5. Train up to EPOCHS with early stopping (PATIENCE=12)
        6. Save best checkpoint by val C-index

    Args:
        fold        : Fold number (1-indexed)
        train_ids   : Patient IDs for training in this fold
        val_ids     : Patient IDs for validation in this fold
        clinical_df : Processed clinical DataFrame
        device      : torch.device

    Returns:
        dict of best val metrics: {"c_index": float, "td_auc": dict, "ibs": float}
    """
    print(f"\n{'='*65}")
    print(f"  FOLD {fold}  |  Train: {len(train_ids)} patients  |  Val: {len(val_ids)} patients")
    print(f"{'='*65}")

    train_index = build_patient_index(DATA_ROOT, train_ids)
    val_index   = build_patient_index(DATA_ROOT, val_ids)

    clinical_feature_cols = [
        c for c in clinical_df.columns
        if c not in ["Patient_ID", "time", "event"]
    ]
    train_dataset = GliomaDataset(patient_index=train_index, clinical_df=clinical_df, augment=True,
                                  clinical_feature_cols=clinical_feature_cols)
    val_dataset = GliomaDataset(
        patient_index=val_index,
        clinical_df=clinical_df,
        augment=False,
        clinical_feature_cols=clinical_feature_cols,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    # Training survival arrays for IPCW correction in td-AUC/IBS
    train_times  = clinical_df[clinical_df["Patient_ID"].isin(train_ids)]["time"].values
    train_events = clinical_df[clinical_df["Patient_ID"].isin(train_ids)]["event"].values

    # Fresh model + optimizer + scheduler for every fold
    model     = GliomaModel(num_clinical_features=len(clinical_feature_cols)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    best_c_index      = 0.0
    best_metrics      = {}
    best_ckpt_path    = CHECKPOINT_DIR / f"fold_{fold}_best.pt"
    epochs_no_improve = 0

    print(
        f"\n{'Epoch':<8} {'LR':<10} {'Train Loss':<14} {'C-index':<12} "
        f"{'IBS':<10} {'AUC@30d':<12} {'AUC@6mo':<12}"
    )
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics    = validate(model, val_loader, device, train_times, train_events)
        scheduler.step()

        c_idx   = metrics.get("c_index", float("nan"))
        ibs     = metrics.get("ibs",     float("nan"))
        td_auc  = metrics.get("td_auc",  {})
        auc_30  = td_auc.get(30,  float("nan"))
        auc_180 = td_auc.get(180, float("nan"))

        print(
            f"\n{epoch:<8d} {current_lr:<10.2e} {train_loss:<14.4f} {c_idx:<12.4f} "
            f"{ibs:<10.4f} {auc_30:<12.4f} {auc_180:<12.4f}"
        )

        if c_idx > best_c_index:
            best_c_index      = c_idx
            best_metrics      = metrics.copy()
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, c_idx, best_ckpt_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(
                f"\n  [INFO] Early stopping at epoch {epoch} "
                f"(no improvement for {PATIENCE} consecutive epochs)."
            )
            break

    print(f"\n  Fold {fold} best C-index: {best_c_index:.4f}")
    return best_metrics


# ===========================================================================
# BRESLOW ESTIMATOR
# ===========================================================================
def fit_breslow_estimator(
    model: torch.nn.Module,
    all_ids: list,
    clinical_df: pd.DataFrame,
    clinical_feature_cols: list,
    device: torch.device,
) -> dict:
    """
    Fit Breslow baseline cumulative hazard estimator
    using multimodal MRI + clinical risk predictions.
    """

    print("\n[INFO] Fitting Breslow estimator on full dataset ...")

    model.eval()

    # ------------------------------------------------------------
    # Build full dataset
    # ------------------------------------------------------------
    all_index = build_patient_index(
        DATA_ROOT,
        all_ids
    )

    all_dataset = GliomaDataset(
        all_index,
        clinical_df,
        augment=False,
        clinical_feature_cols=clinical_feature_cols,
    )

    all_loader = DataLoader(
        all_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ------------------------------------------------------------
    # Collect risks and survival outcomes
    # ------------------------------------------------------------
    all_risk = []
    all_time = []
    all_event = []

    with torch.no_grad():

        for x, seq_mask, clinical_feat, time, event in all_loader:

            x = x.to(device)

            seq_mask = seq_mask.to(device)

            clinical_feat = clinical_feat.to(device)

            # ----------------------------------------------------
            # Multimodal risk prediction
            # ----------------------------------------------------
            risk = model(
                x,
                seq_mask,
                clinical_feat
            ).squeeze(1)

            all_risk.extend(
                risk.cpu().numpy().tolist()
            )

            all_time.extend(
                time.numpy().tolist()
            )

            all_event.extend(
                event.numpy().tolist()
            )

    # ------------------------------------------------------------
    # Convert to numpy
    # ------------------------------------------------------------
    risks = np.array(all_risk, dtype=np.float64)

    times = np.array(all_time, dtype=np.float64)

    events = np.array(all_event, dtype=np.float64)

    # ------------------------------------------------------------
    # Center risks for numerical stability
    # ------------------------------------------------------------
    mean_risk = risks.mean()

    risks_c = risks - mean_risk

    # ------------------------------------------------------------
    # Sort by survival time
    # ------------------------------------------------------------
    sort_idx = np.argsort(times)

    risks_s = risks_c[sort_idx]

    times_s = times[sort_idx]

    events_s = events[sort_idx]

    # ------------------------------------------------------------
    # Exponentiated risks
    # ------------------------------------------------------------
    exp_risks = np.exp(risks_s)

    # ------------------------------------------------------------
    # Breslow cumulative hazard estimation
    # ------------------------------------------------------------
    breslow_times = []

    breslow_cumhaz = []

    cumhaz = 0.0

    for i in range(len(times_s)):

        # Only event patients contribute hazard jumps
        if events_s[i] == 1:

            # Patients still at risk
            risk_set_sum = exp_risks[i:].sum()

            if risk_set_sum > 0:

                # Hazard increment
                cumhaz += 1.0 / risk_set_sum

            breslow_times.append(times_s[i])

            breslow_cumhaz.append(cumhaz)

    print(
        f"[INFO] Breslow estimator fitted: "
        f"{len(risks)} patients | "
        f"{int(events.sum())} events | "
        f"{len(breslow_times)} unique event times."
    )

    return {
        "times": np.array(breslow_times),

        "cumhazard": np.array(breslow_cumhaz),

        "mean_risk": mean_risk,
    }

# ===========================================================================
# INFERENCE UTILITIES
# ===========================================================================

def progression_probability(
    log_risk:      float,
    breslow_table: dict,
    days:          list,
) -> dict:
    """
    Convert a model log-risk score to P(progression within X days).

    Formula:
        S(t) = exp( -H0(t) x exp(r - mean_risk) )
        P(t) = 1 - S(t)

    Args:
        log_risk      : Raw scalar from model.forward().squeeze().item()
        breslow_table : dict from fit_breslow_estimator() or loaded .pkl
        days          : List of time horizons in days

    Returns:
        {days: probability}  e.g. {30: 0.08, 180: 0.41}
    """
    centred_risk = log_risk - breslow_table["mean_risk"]
    exp_risk     = np.exp(centred_risk)
    breslow_t    = breslow_table["times"]
    breslow_h0   = breslow_table["cumhazard"]

    result = {}
    for t in days:
        idx = np.searchsorted(breslow_t, t, side="right") - 1
        h0  = breslow_h0[idx] if idx >= 0 else 0.0
        p   = 1.0 - np.exp(-h0 * exp_risk)
        result[t] = float(np.clip(p, 0.0, 1.0))
    return result


def predict_patient(
    patient_dir:   Path,
    clinical_df: pd.DataFrame,
    clinical_feature_cols : list,
    model:         torch.nn.Module,
    breslow_table: dict,
    device:        torch.device,
    time_horizons: list = [30, 180, 365],
) -> dict:
    """
    Predict tumor progression probabilities for a single new patient.

    Loads MRI sequence from patient_dir, runs model inference, converts
    log-risk to P(progression within X days) via Breslow estimator.

    Output interpretation:
        P near 1.0 = model is confident progression is imminent
        P near 0.0 = model predicts stable disease

    Args:
        patient_dir    : Path to patient directory (contains Timepoint_X/ subfolders)
        model          : Trained GliomaModel (from load_for_inference)
        breslow_table  : dict from fit_breslow_estimator (from load_for_inference)
        device         : torch.device
        time_horizons  : Days to evaluate progression probability at

    Returns:
        dict:
            "log_risk"      : raw model score (float)
            "probabilities" : {days: probability}
            "interpretation": human-readable summary with progress bars

    Example:
        model, breslow = load_for_inference(ckpt_path, breslow_path, device)
        result = predict_patient(
            patient_dir   = Path("MU-Glioma-Post/PatientID_0001"),
            model         = model,
            breslow_table = breslow,
            device        = device,
            time_horizons = [30, 180, 365],
        )
        print(result["interpretation"])
    """
    model.eval()
    patient_id = patient_dir.name

    timepoints = sorted([
        patient_dir / tp
        for tp in os.listdir(patient_dir)
        if (patient_dir / tp).is_dir()
    ])
    if not timepoints:
        raise ValueError(f"No timepoint subdirectories found in: {patient_dir}")

    sequence, seq_mask = build_patient_sequence(
        tp_paths=timepoints, max_timepoints=MAX_TIMEPOINTS, augment_fn=None
    )
    clinical_row = clinical_df[
        clinical_df["Patient_ID"] == patient_id
        ]

    if len(clinical_row) == 0:
        raise ValueError(f"No clinical row found for patient: {patient_id}")

    clinical_row = clinical_row.iloc[0]

    clinical_feat = clinical_row[
            clinical_feature_cols
        ].values.astype(np.float32)
    
    x_t    = torch.tensor(sequence,  dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(seq_mask,  dtype=torch.float32).unsqueeze(0).to(device)
    clinical_t = torch.tensor(
        clinical_feat,
        dtype = torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        log_risk = model(x_t, mask_t, clinical_t).squeeze().item()

    probs = progression_probability(log_risk, breslow_table, days=time_horizons)

    lines = [f"Patient : {patient_dir.name}", f"Log-risk: {log_risk:.4f}", ""]
    for t, p in probs.items():
        label = f"{t}d" if t < 365 else f"{t//365}yr"
        bar   = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        lines.append(f"  P(progression <= {label:>4s}): {p:5.1%}  [{bar}]")

    return {
        "log_risk"      : log_risk,
        "probabilities" : probs,
        "interpretation": "\n".join(lines),
    }


def load_for_inference(
    checkpoint_path: Path,
    breslow_path:    Path,
    num_clinical_features,
    device:          torch.device,
) -> tuple:
    """
    Load a trained model and Breslow table from disk for inference.

    Args:
        checkpoint_path : .pt checkpoint file (saved by save_checkpoint)
        breslow_path    : .pkl Breslow table (saved by run_training)
        device          : torch.device

    Returns:
        (model, breslow_table) — pass directly to predict_patient()
    """
    model = GliomaModel(num_clinical_features=num_clinical_features).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[INFO] Model loaded  : {checkpoint_path}")
    print(f"       Epoch {ckpt['epoch']}  |  C-index: {ckpt['c_index']:.4f}")

    with open(breslow_path, "rb") as f:
        breslow_table = pickle.load(f)
    print(f"[INFO] Breslow loaded: {breslow_path}")
    print(f"       Event times: {len(breslow_table['times'])}  "
          f"|  Mean risk: {breslow_table['mean_risk']:.4f}")

    return model, breslow_table


# ===========================================================================
# MAIN TRAINING FUNCTION
# ===========================================================================

def run_training():
    """
    Full training pipeline:

        1. Load clinical data
        2. 85/15 patient split -> lock test set away immediately
        3. 5-fold CV on the 85% pool: train + validate per fold
        4. Aggregate and print CV metrics
        5. Evaluate best fold model on locked test set (once only)
        6. Fit Breslow estimator on ALL patients using best model
        7. Save Breslow table + print inference instructions

    Returns:
        (fold_results, breslow_table)
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  Glioma Spatiotemporal Survival Model - Training")
    print(f"  Device        : {device}")
    print(f"  Epochs        : {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"  Warmup epochs : {WARMUP_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  LR            : {LEARNING_RATE}  |  Weight decay: {WEIGHT_DECAY}")
    print(f"  CV folds      : {N_FOLDS}  |  Test fraction: {TEST_FRACTION:.0%}")
    print(f"  AUC horizons  : {AUC_TIME_HORIZONS} days")
    print("=" * 65 + "\n")

    # 1. Clinical data
    clinical_df = load_or_build_clinical()
    clinical_feature_cols = [
        c for c in clinical_df.columns if c not in ["Patient_ID", "time", "event:"]
    ]

    # 2. MRI patient list
    all_patients = sorted([
        p for p in os.listdir(DATA_ROOT) if (DATA_ROOT / p).is_dir()
    ])
    print(f"[INFO] Found {len(all_patients)} patient directories in MRI root.\n")

    # 3. 85/15 split
    cv_ids, test_ids = split_patients_85_15(all_patients)

    # 4. 5-fold CV
    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_arr       = np.array(cv_ids)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_arr), start=1):
        train_ids    = cv_arr[train_idx].tolist()
        val_ids      = cv_arr[val_idx].tolist()
        fold_metrics = train_fold(fold, train_ids, val_ids, clinical_df, device)
        fold_results.append(fold_metrics)

    # 5. CV summary
    print("\n" + "=" * 65)
    print(f"  {N_FOLDS}-FOLD CROSS VALIDATION SUMMARY")
    print("=" * 65)

    c_indices  = [r.get("c_index", float("nan")) for r in fold_results]
    ibs_scores = [r.get("ibs",     float("nan")) for r in fold_results]

    print(f"\n  Discrimination (C-index):")
    print(f"    Mean +/- Std : {np.nanmean(c_indices):.4f} +/- {np.nanstd(c_indices):.4f}")
    print(f"    Median       : {np.nanmedian(c_indices):.4f}")
    print(f"    Per fold     : {[round(c, 4) for c in c_indices]}")

    print(f"\n  Calibration (IBS — lower = better, null model ~0.25):")
    print(f"    Mean +/- Std : {np.nanmean(ibs_scores):.4f} +/- {np.nanstd(ibs_scores):.4f}")

    print(f"\n  Time-dependent AUC:")
    for t in AUC_TIME_HORIZONS:
        label = "30 days (1 month)" if t == 30 else "180 days (6 months)"
        aucs  = [r.get("td_auc", {}).get(t, float("nan")) for r in fold_results]
        aucs  = [a for a in aucs if not np.isnan(a)]
        if aucs:
            print(f"    AUC @ {label}: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        else:
            print(f"    AUC @ {label}: not computable (horizons outside event range)")

    # 6. Held-out test set evaluation
    best_fold = int(np.nanargmax(c_indices)) + 1
    best_ckpt = CHECKPOINT_DIR / f"fold_{best_fold}_best.pt"

    print(f"\n[INFO] Best fold: {best_fold} (C-index={c_indices[best_fold-1]:.4f})")
    print(f"[INFO] Evaluating on held-out test set ({len(test_ids)} patients) ...")

    test_index   = build_patient_index(DATA_ROOT, test_ids)
    test_dataset = GliomaDataset(test_index, clinical_df, augment=False, clinical_feature_cols=clinical_feature_cols)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    final_model = GliomaModel(num_clinical_features=len(clinical_feature_cols)).to(device)
    final_model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])

    cv_times  = clinical_df[clinical_df["Patient_ID"].isin(cv_ids)]["time"].values
    cv_events = clinical_df[clinical_df["Patient_ID"].isin(cv_ids)]["event"].values

    test_metrics = validate(final_model, test_loader, device, cv_times, cv_events)

    print("\n" + "=" * 65)
    print("  HELD-OUT TEST SET RESULTS  (unbiased estimate, reported once)")
    print("=" * 65)
    print(f"  C-index : {test_metrics.get('c_index', float('nan')):.4f}")
    print(f"  IBS     : {test_metrics.get('ibs',     float('nan')):.4f}")
    for t, auc in test_metrics.get("td_auc", {}).items():
        label = "30d" if t == 30 else "6mo"
        print(f"  AUC@{label}  : {auc:.4f}")
    print("=" * 65)

    # 7. Breslow estimator on ALL patients
    breslow_table = fit_breslow_estimator(
        model=final_model, all_ids=all_patients,
        clinical_df=clinical_df,
         clinical_feature_cols=clinical_feature_cols, device=device,
    )

    breslow_path = CHECKPOINT_DIR / "breslow_table.pkl"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(breslow_path, "wb") as f:
        pickle.dump(breslow_table, f)
    print(f"[INFO] Breslow table saved -> {breslow_path}")

    # 8. Inference instructions
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE - HOW TO RUN INFERENCE ON A NEW PATIENT")
    print("=" * 65)
    print(f"""
    from train import load_for_inference, predict_patient
    from pathlib import Path
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, breslow = load_for_inference(
        checkpoint_path = Path(r"{best_ckpt}"),
        breslow_path    = Path(r"{breslow_path}"),
        device          = device,
    )

    result = predict_patient(
        patient_dir   = Path(r"MU-Glioma-Post\\PatientID_XXXX"),
        model         = model,
        breslow_table = breslow,
        device        = device,
        time_horizons = [30, 180, 365],
    )
    print(result["interpretation"])
    """)

    return fold_results, breslow_table


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    run_training()
