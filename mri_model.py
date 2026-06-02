"""
mri_model.py
============
All MRI-related processing, the PyTorch Dataset, the model architecture,
and the Cox loss for the glioma spatiotemporal survival pipeline.

Directory structure assumed:
    MU-Glioma-Post/
        PatientID_XXXX/
            Timepoint_X/
                brain_t1n*.nii.gz      ← T1 native
                brain_t1c*.nii.gz      ← T1 contrast-enhanced
                brain_t2w*.nii.gz      ← T2-weighted
                brain_t2f*.nii.gz      ← FLAIR
                tumorMask.nii          ← segmentation mask (labels 1-4)

MRI -> CNN -> LSTM __
                     |
                     |---> Fusion -> Prediction Head -> Risk
Clinical -> MLP _____|

Contents:
  § 1  MRI loading utilities
  § 2  Intensity normalisation + mask decomposition
  § 3  Tumour-centred 3-D patch extraction + padding
  § 4  Per-patient longitudinal sequence builder
  § 5  MONAI augmentation transforms
  § 6  GliomaDataset  (PyTorch Dataset)
  § 7  Model: MobileNetEncoder → TemporalLSTM → PredictionHead → GliomaModel
  § 8  Cox Proportional Hazards loss
  § 9  build_patient_index  (utility for train.py)
"""

import torchvision.models as models
import os
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

from monai.transforms import (
    Compose,
    RandGaussianNoise,
    RandScaleIntensity,
)

warnings.filterwarnings("ignore")


# ===========================================================================
# CONSTANTS / HYPERPARAMETERS
# ===========================================================================

PATCH_SIZE     = 64    # Isotropic spatial size of every extracted 3-D patch (voxels).
                       # All patches are resized to (64, 64, 64) regardless of
                       # original tumor size. Larger = more detail, more memory.

PATCH_MARGIN   = 15    # Extra voxels added around the tight tumor bounding box
                       # on every side. Captures peritumoral tissue (edema, infiltration)
                       # that may hold progression-relevant signal.

MAX_TIMEPOINTS = 6     # Maximum number of longitudinal scans per patient.
                       # Sequences shorter than this are zero-padded.
                       # Set by the dataset: up to 6 MRI timepoints per patient.

IN_CHANNELS    = 8     # Number of input channels per 3-D patch:
                       #   4 MRI modalities : T1, T1c, T2, FLAIR
                       #   4 mask channels  : NETC, Edema, Enhancing Tumor, Resection Cavity

CNN_OUT_DIM    = 512   # Output dimension of the MobileNet encoder per timepoint.
                       # This is the spatial feature vector fed into the LSTM.

LSTM_HIDDEN    = 256   # Number of hidden units in the LSTM.
                       # Controls how much temporal context the model can store.

SLICE_CHUNK    = 16    # Number of 2-D slices processed by MobileNet in one sub-batch.
                       # Without chunking, a batch of B=4 patients × T=6 timepoints
                       # × D=64 slices = 1536 slices hit the GPU at once → OOM on 8GB.
                       # Chunking processes SLICE_CHUNK slices at a time, accumulating
                       # features, then concatenating — identical result, fraction of peak
                       # VRAM. Tune down if still OOM, up if you have headroom.

# Zero-filled placeholder patch used when a timepoint is missing or corrupted.
# Shape matches a real patch: (IN_CHANNELS, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
ZERO_SCAN = np.zeros((IN_CHANNELS, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


# ===========================================================================
# § 1 — MRI LOADING UTILITIES
# ===========================================================================

def find_file(keyword: str, files: list, tp_path: Path):
    """
    Search a list of filenames for the first one containing `keyword`
    (case-insensitive match) and return its full Path.

    Returns None (instead of raising) when no match is found, so the
    caller can handle missing modalities gracefully without crashing.

    Args:
        keyword : Substring to search for (e.g. "t1n", "t2f", "tumormask")
        files   : List of filename strings from os.listdir()
        tp_path : Parent directory Path — prepended to the matched filename

    Returns:
        Full Path to matched file, or None if not found

    Example:
        find_file("t1n", ["brain_t1n_stripped.nii.gz", "tumorMask.nii"], tp_path)
        → tp_path / "brain_t1n_stripped.nii.gz"
    """
    keyword_lower = keyword.lower()
    for f in files:
        if keyword_lower in f.lower():
            return tp_path / f
    return None


def load_timepoint(tp_path: Path):
    """
    Load all four MRI modalities and the tumour segmentation mask from a
    single timepoint directory.

    File-name keywords (matched case-insensitively via find_file):
        "t1n"       → T1 native          (brain_t1n*.nii.gz)
        "t1c"       → T1 contrast        (brain_t1c*.nii.gz)
        "t2w"       → T2-weighted        (brain_t2w*.nii.gz)
        "t2f"       → FLAIR              (brain_t2f*.nii.gz)
        "tumormask" → segmentation mask  (tumorMask.nii)

    If any required file is missing, the entire timepoint is skipped
    (returns None) rather than loading an incomplete or misaligned set.

    Args:
        tp_path : Path to the timepoint directory (e.g. .../PatientID_0001/Timepoint_2/)

    Returns:
        Tuple (t1, t1c, t2, flair, mask) as float32 numpy arrays, each with
        the same spatial shape (X, Y, Z). Returns None if any file is missing.
    """
    try:
        files = os.listdir(tp_path)
    except FileNotFoundError:
        return None   # Timepoint directory does not exist on disk

    paths = {
        "t1"    : find_file("t1n",       files, tp_path),
        "t1c"   : find_file("t1c",       files, tp_path),
        "t2"    : find_file("t2w",       files, tp_path),
        "flair" : find_file("t2f",       files, tp_path),
        "mask"  : find_file("tumormask", files, tp_path),  # matches "tumorMask"
    }

    # Skip this timepoint entirely if any file is absent
    missing = [k for k, v in paths.items() if v is None]
    if missing:
        return None

    # Load NIfTI volumes and cast to float32 for PyTorch compatibility
    t1    = nib.load(paths["t1"]).get_fdata().astype(np.float32)
    t1c   = nib.load(paths["t1c"]).get_fdata().astype(np.float32)
    t2    = nib.load(paths["t2"]).get_fdata().astype(np.float32)
    flair = nib.load(paths["flair"]).get_fdata().astype(np.float32)
    mask  = nib.load(paths["mask"]).get_fdata().astype(np.float32)

    return t1, t1c, t2, flair, mask


# ===========================================================================
# § 2 — INTENSITY NORMALISATION + MASK DECOMPOSITION
# ===========================================================================

def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalise an MRI volume to zero mean and unit standard deviation.

    Why z-score?
        MRI intensities are not standardised across scanners or sessions.
        Raw values depend on scanner settings and are not comparable between
        patients. Z-scoring removes scanner-specific scale and offset, making
        intensity patterns comparable across patients and timepoints.

    Edge case: if std ≈ 0 (blank or near-constant volume), return unchanged
    to avoid division-by-zero. This can happen for missing/empty scans.

    Args:
        volume : (X, Y, Z) float32 numpy array — raw MRI voxel intensities

    Returns:
        (X, Y, Z) float32 array — normalised intensities (mean≈0, std≈1)
    """
    volume = volume.astype(np.float32)
    std = volume.std()
    if std < 1e-8:
        return volume   # Blank volume — return as-is
    return (volume - volume.mean()) / std


def split_mask(mask: np.ndarray) -> tuple:
    """
    Decompose a multi-label segmentation mask into 4 separate binary channels.

    The tumor mask uses integer labels defined by the dataset:
        Label 1 → Non-enhancing tumor core (NETC) / Necrotic core
        Label 2 → Tumor infiltration / edema (peritumoral zone)
        Label 3 → Enhancing tumor tissue (active tumor)
        Label 4 → Resection cavity (surgically removed region)

    Why separate channels?
        The model needs to learn how each sub-region evolves independently
        over time. A necrotic core growing means different things than
        an enhancing region growing. Stacking as binary channels lets the
        CNN treat each sub-region as a distinct spatial feature map.

    Args:
        mask : (X, Y, Z) float32 array with integer labels 0–4

    Returns:
        Tuple of 4 binary (0.0 / 1.0) float32 arrays, each shape (X, Y, Z):
            (necrotic, edema, enhancing, cavity)
    """
    return (
        (mask == 1).astype(np.float32),   # Necrotic core
        (mask == 2).astype(np.float32),   # Edema / infiltration
        (mask == 3).astype(np.float32),   # Enhancing tumor
        (mask == 4).astype(np.float32),   # Resection cavity
    )


# ===========================================================================
# § 3 — TUMOUR-CENTRED 3-D PATCH EXTRACTION + RESIZING
# ===========================================================================

def get_tumor_bbox(mask: np.ndarray):
    """
    Compute the tight axis-aligned bounding box around all non-zero voxels
    in the segmentation mask.

    Args:
        mask : (X, Y, Z) float32 or int array — tumor labels (any non-zero = tumor)

    Returns:
        Tuple (x_min, x_max, y_min, y_max, z_min, z_max) as ints,
        or None if the mask is entirely zero (no tumor found).
    """
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None   # No tumor voxels found
    return (
        int(coords[0].min()), int(coords[0].max()),
        int(coords[1].min()), int(coords[1].max()),
        int(coords[2].min()), int(coords[2].max()),
    )


def get_tumor_center(bbox: tuple) -> tuple:
    """
    Return the integer voxel-space centre of a bounding box.

    Args:
        bbox : (x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        (cx, cy, cz) — integer centre coordinates
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return (
        (x_min + x_max) // 2,
        (y_min + y_max) // 2,
        (z_min + z_max) // 2,
    )


def resize_patch(patch: np.ndarray, target_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Resize a variable-size 3-D patch to a fixed isotropic size using
    channel-appropriate interpolation methods.

    Interpolation strategy:
        MRI channels (0–3) : Trilinear — preserves smooth intensity gradients
        Mask channels (4–7): Nearest neighbour — preserves binary 0/1 values
                             (trilinear would blur boundaries and create
                             fractional values like 0.3, 0.7 which are wrong
                             for binary segmentation masks)

    Args:
        patch       : (C, X, Y, Z) float32 numpy array — C=8 channels
        target_size : Output spatial size (isotropic cube), default PATCH_SIZE=64

    Returns:
        (C, target_size, target_size, target_size) float32 numpy array
    """
    target = (target_size, target_size, target_size)

    # Convert numpy → torch tensor, add batch dim: (1, C, X, Y, Z)
    t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)

    # Resize MRI modality channels (0-3) with trilinear interpolation
    mri_resized  = interpolate(t[:, :4], size=target, mode="trilinear", align_corners=False)

    # Resize binary mask channels (4-7) with nearest-neighbour interpolation
    mask_resized = interpolate(t[:, 4:], size=target, mode="nearest")

    # Recombine channels
    resized = torch.cat([mri_resized, mask_resized], dim=1)  # (1, 8, T, T, T)
    return resized.squeeze(0).numpy()                         # (8, T, T, T)


def _pad_to_minimum(patch: np.ndarray, min_size: int) -> np.ndarray:
    """
    Zero-pad a patch along any spatial dimension smaller than min_size.

    Used only in the no-tumor fallback path in extract_full_tumor_patch()
    when a center-crop is smaller than the target size.

    Args:
        patch    : (C, X, Y, Z) numpy array
        min_size : Minimum required spatial size per dimension

    Returns:
        (C, min_size, min_size, min_size) numpy array (may crop if oversized)
    """
    C, X, Y, Z = patch.shape

    def _p(s):
        """Compute symmetric padding for dimension of size s."""
        total = max(min_size - s, 0)
        return total // 2, total - total // 2

    patch = np.pad(patch, ((0, 0), _p(X), _p(Y), _p(Z)), mode="constant")
    return patch[:, :min_size, :min_size, :min_size]


def extract_full_tumor_patch(
    volume:      np.ndarray,
    mask:        np.ndarray,
    margin:      int = PATCH_MARGIN,
    target_size: int = PATCH_SIZE,
) -> np.ndarray:
    """
    Extract the full tumor region from the 8-channel volume with a surrounding
    margin, then resize to target_size³ for consistent model input.

    Unlike a fixed center-crop, this approach guarantees that NO tumor voxels
    are cut off, regardless of tumor size, shape, or location in the brain.
    The margin captures peritumoral tissue (edema, infiltration) that may
    contain progression-relevant signal.

    Process:
        1. Compute tight bounding box around all non-zero mask voxels
        2. Expand by `margin` voxels on each side (clips at brain boundary)
        3. Crop the expanded region from the 8-channel volume
        4. Resize to (target_size, target_size, target_size)

    Fallback (no tumor found):
        Center-crop the brain volume at target_size/2 radius, pad if needed,
        then resize. This handles timepoints with missing/empty masks.

    Args:
        volume      : (8, X, Y, Z) float32 — stacked MRI + mask channels
        mask        : (X, Y, Z) float32 — original multi-label mask (for bbox only)
        margin      : Extra voxels around bbox on each side (default: PATCH_MARGIN=15)
        target_size : Output spatial size after resizing (default: PATCH_SIZE=64)

    Returns:
        (8, target_size, target_size, target_size) float32 numpy array
    """
    _, X, Y, Z = volume.shape

    bbox = get_tumor_bbox(mask)

    if bbox is None:
        # No tumor — fall back to center crop of the brain
        cx, cy, cz = X // 2, Y // 2, Z // 2
        half = target_size // 2
        crop = volume[
            :,
            max(cx - half, 0):min(cx + half, X),
            max(cy - half, 0):min(cy + half, Y),
            max(cz - half, 0):min(cz + half, Z),
        ]
        crop = _pad_to_minimum(crop, target_size)
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = bbox

        # Expand bounding box by margin, clamped to image boundaries
        x1 = max(x_min - margin, 0);  x2 = min(x_max + margin, X)
        y1 = max(y_min - margin, 0);  y2 = min(y_max + margin, Y)
        z1 = max(z_min - margin, 0);  z2 = min(z_max + margin, Z)

        crop = volume[:, x1:x2, y1:y2, z1:z2]

    # Resize to fixed target size using modality-appropriate interpolation
    return resize_patch(crop, target_size)


# ===========================================================================
# § 4 — LONGITUDINAL SEQUENCE BUILDER
# ===========================================================================

def build_patient_sequence(
    tp_paths:       list,
    max_timepoints: int  = MAX_TIMEPOINTS,
    augment_fn      = None,
) -> tuple:
    """
    Build the padded longitudinal sequence representation for one patient.

    For each valid timepoint directory in chronological order:
        1. Load 4 MRI modalities + tumor mask (skip if any file missing)
        2. Z-score normalise each modality independently
        3. Decompose mask → 4 binary channels (NETC, Edema, ET, RC)
        4. Stack → 8-channel 3-D tensor of shape (8, X, Y, Z)
        5. Extract tumor-centred patch → resized to (8, 64, 64, 64)
        6. Optionally apply augmentation (training only)

    Padding:
        If a patient has fewer than max_timepoints valid scans, the sequence
        is right-padded with ZERO_SCAN (all zeros). The seq_mask tensor
        records which positions are real (1.0) vs padded (0.0) so the LSTM
        can ignore padding during temporal modeling.

    Truncation:
        If a patient has more than max_timepoints scans (shouldn't happen
        with this dataset, max=6), we silently stop after max_timepoints.

    Args:
        tp_paths       : Sorted list of Path objects, one per timepoint dir
        max_timepoints : Sequence length (pad or truncate to this), default 6
        augment_fn     : Optional MONAI Compose transform (train only)

    Returns:
        scans    : np.ndarray shape (max_timepoints, 8, 64, 64, 64) float32
                   — padded sequence of tumor patches
        seq_mask : np.ndarray shape (max_timepoints,) float32
                   — 1.0 for real scans, 0.0 for padding positions
    """
    scans = []

    for tp_path in tp_paths:
        data = load_timepoint(tp_path)
        if data is None:
            continue   # Skip corrupted or incomplete timepoints

        t1, t1c, t2, flair, mask = data

        # --- Normalise each MRI modality independently ---
        t1, t1c, t2, flair = (
            normalize_mri(t1),
            normalize_mri(t1c),
            normalize_mri(t2),
            normalize_mri(flair),
        )

        # --- Decompose multi-label mask into 4 binary channels ---
        necrotic, edema, enhancing, cavity = split_mask(mask)

        # --- Stack into 8-channel tensor: (8, X, Y, Z) ---
        tensor = np.stack(
            [t1, t1c, t2, flair, necrotic, edema, enhancing, cavity],
            axis=0,
        )

        # --- Extract tumor-centred patch and resize to (8, 64, 64, 64) ---
        patch = extract_full_tumor_patch(tensor, mask)

        # --- Optionally augment (training only) ---
        if augment_fn is not None:
            patch = augment_fn(patch)
            if isinstance(patch, torch.Tensor):
                patch = patch.numpy()

        scans.append(patch.astype(np.float32))

        # Stop once maximum sequence length is reached
        if len(scans) == max_timepoints:
            break

    # --- Handle edge case: all timepoints failed to load ---
    if len(scans) == 0:
        scans = [ZERO_SCAN.copy()]   # At least one zero-filled scan

    # --- Build sequence mask: 1.0=real scan, 0.0=padding ---
    n_real   = len(scans)
    seq_mask = [1.0] * n_real + [0.0] * (max_timepoints - n_real)

    # --- Right-pad with zero scans to reach max_timepoints ---
    while len(scans) < max_timepoints:
        scans.append(ZERO_SCAN.copy())

    return (
        np.stack(scans, axis=0).astype(np.float32),   # (T, 8, 64, 64, 64)
        np.array(seq_mask, dtype=np.float32),          # (T,)
    )


# ===========================================================================
# § 5 — MONAI AUGMENTATION (training only)
# ===========================================================================

# Augmentation pipeline applied only during training (augment=True in Dataset).
# Augmentations are mild to avoid distorting clinically meaningful signals:
#   - RandGaussianNoise : simulates scanner noise (prob=0.2, std=0.05)
#   - RandScaleIntensity: simulates scanner gain variation (prob=0.3, ±10%)
#
# Geometric augmentations (flip, rotate) are commented out because flipping
# left/right or rotating a brain scan may destroy lateralisation information
# that could be clinically relevant for tumor progression.
TRAIN_AUGMENT = Compose([
    # RandFlip(prob=0.5, spatial_axis=0),    # Disabled: may flip lateralisation
    # RandFlip(prob=0.5, spatial_axis=1),    # Disabled: may flip lateralisation
    # RandRotate90(prob=0.5),                # Disabled: may alter anatomy
    RandGaussianNoise(prob=0.2, std=0.05),   # Simulate MRI scanner noise
    RandScaleIntensity(prob=0.3, factors=0.1),  # Simulate scanner gain variation
])


# ===========================================================================
# § 6 — PYTORCH DATASET
# ===========================================================================


class GliomaDataset(Dataset):
    """
    Dataset for longitudinal glioma MRI survival modeling.
    PyTorch Dataset for the MU-Glioma-Post longitudinal MRI dataset.

    Combines:
        - MRI data from the patient_index (built by build_patient_index)
        - Survival labels from clinical_df (output of preprocess_clinical)

    Only patients present in BOTH sources are included. Patients present
    in MRI index but missing from clinical CSV are silently dropped.

    Each __getitem__ call returns a single patient's data:
        sequence  : Tensor (MAX_TIMEPOINTS, 8, 64, 64, 64)
                    Padded longitudinal sequence of tumor patches.
                    Missing timepoints are zero-filled.
        seq_mask  : Tensor (MAX_TIMEPOINTS,)
                    1.0 = real scan, 0.0 = padding
        clinical_feat
        time      : Tensor scalar — days to event or censoring
        event     : Tensor scalar — 1.0 = progression, 0.0 = censored

    Args:
        patient_index : dict {patient_id (str) → [Path, ...]}
                        Built by build_patient_index().
        clinical_df   : DataFrame with columns [Patient_ID, event, time, ...]
                        Output of clinical_preprocessing.preprocess_clinical().
        augment       : bool — if True, apply TRAIN_AUGMENT per patch.
                        Should be True for train set, False for val/test.
        clinical_feature_cols
    Returns:
        sequence        : (T, C, D, H, W)
        seq_mask        : (T,)
        clinical_feat   : (num_clinical_features,)
        time            : scalar survival time
        event           : scalar event indicator
    """

    def __init__(
        self,
        patient_index,
        clinical_df,
        augment=False,
        clinical_feature_cols=None,
    ):
        self.patient_index = patient_index
        self.clinical_df = clinical_df
        self.augment = augment
        self.clinical_feature_cols = clinical_feature_cols

        if self.clinical_feature_cols is None:
            self.clinical_feature_cols = [
                c for c in clinical_df.columns
                if c not in ["Patient_ID", "time", "event"]
            ]

        self.patient_ids = list(patient_index.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        tp_paths = self.patient_index[patient_id]

        sequence, seq_mask = build_patient_sequence(
            tp_paths=tp_paths,
            max_timepoints=MAX_TIMEPOINTS,
            augment_fn=TRAIN_AUGMENT if self.augment else None,
        )

        clinical_row = self.clinical_df[
            self.clinical_df["Patient_ID"] == patient_id
        ].iloc[0]

        clinical_feat = clinical_row[self.clinical_feature_cols].values.astype(np.float32)

        time = np.float32(clinical_row["time"])
        event = np.float32(clinical_row["event"])

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(seq_mask, dtype=torch.float32),
            torch.tensor(clinical_feat, dtype=torch.float32),
            torch.tensor(time, dtype=torch.float32),
            torch.tensor(event, dtype=torch.float32),
        )
    
class ClinicalEncoder(nn.Module):
    """
    Encodes tabular clinical features into a dense embedding.

    Input:
        (B, num_clinical_features)

    Output:
        (B, clinical_embedding_dim)

    Purpose:
        Transform heterogeneous clinical variables into a compact
        latent representation that can be fused with MRI features.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.encoder(x)


# ===========================================================================
# § 7 — MODEL ARCHITECTURE
# ===========================================================================

class MobileNetEncoder(nn.Module):
    """
    2-D MobileNetV3-Small encoder used to extract spatial features from
    individual axial slices of a 3-D MRI patch.

    Why MobileNetV3-Small?
        - Lightweight: designed for efficiency with minimal parameters
        - Strong feature extractor for 2D images via depthwise separable convs
        - The 3-D volume is processed slice-by-slice (axial plane), so a 2-D
          backbone is appropriate and computationally feasible

    Adaptation for MRI:
        - The first Conv2d layer is replaced to accept IN_CHANNELS=8 inputs
          instead of the default 3 (RGB).
        - The ImageNet classification head is removed (replaced with Identity).
        - A new projection head maps backbone features → CNN_OUT_DIM.

    Input  : (B, 8, H, W)  — batch of 2-D slices with 8 channels
    Output : (B, CNN_OUT_DIM)  — spatial feature vector per slice

    Args:
        in_channels : Number of input channels (default: IN_CHANNELS=8)
        out_dim     : Output embedding dimension (default: CNN_OUT_DIM=512)
    """

    def __init__(self, in_channels: int = IN_CHANNELS , out_dim: int = CNN_OUT_DIM):
        super().__init__()

        # Load MobileNetV3-Small with pretrained ImageNet weights
        self.backbone = models.mobilenet_v3_small(pretrained=True)

        # Replace the first convolution to accept 8-channel MRI input
        # (Original expects 3 channels for RGB)
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Remove the ImageNet classification head; we add our own
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        # Projection head: backbone_features → CNN_OUT_DIM
        self.fc = nn.Sequential(
            nn.Linear(num_features, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 8, H, W) — batch of 2-D axial MRI slices

        Returns:
            (B, CNN_OUT_DIM) — spatial feature vector per slice
        """
        # Resize to 224×224 as required by MobileNetV3 pretrained weights
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
        feat = self.backbone(x)
        return self.fc(feat)


class TemporalLSTM(nn.Module):
    """
    Single-layer LSTM that models tumor evolution across longitudinal timepoints.

    The LSTM receives a sequence of spatial feature vectors (one per timepoint)
    and learns temporal patterns such as:
        - Tumor growth acceleration
        - Stabilisation after treatment
        - Emergence of new enhancing regions
        - Pattern shifts between sub-regions over time

    Input  : (B, T, CNN_OUT_DIM) — sequence of per-timepoint feature vectors
    Output : (B, LSTM_HIDDEN)    — final hidden state (summary of full sequence)

    The final hidden state h_n captures the patient's full temporal trajectory
    and is passed to the prediction head for survival risk estimation.

    Args:
        input_size     : Dimension of each input step (= CNN_OUT_DIM)
        hidden_size    : LSTM hidden state dimension (= LSTM_HIDDEN)
        num_layers     : Number of stacked LSTM layers (1 is sufficient here)
        input_dropout  : Dropout applied to input sequence before LSTM
    """

    def __init__(
        self,
        input_size:    int   = CNN_OUT_DIM,
        hidden_size:   int   = LSTM_HIDDEN,
        num_layers:    int   = 1,
        input_dropout: float = 0.2,
    ):
        super().__init__()
        self.input_drop = nn.Dropout(p=input_dropout)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,    # Input shape: (B, T, F) not (T, B, F)
            dropout     = 0.0,     # Inter-layer dropout only useful for num_layers > 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, input_size) — sequence of spatial feature vectors

        Returns:
            (B, hidden_size) — final LSTM hidden state (temporal summary)
        """
        x = self.input_drop(x)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # Take the last layer's hidden state




class PredictionHead(nn.Module):
    """
    Fully-connected head that maps LSTM output → scalar log-risk score.

    The output is a raw unbounded scalar (no sigmoid/softmax/exp applied).
    This raw score is the log-risk value fed directly into the Cox PH loss.

    In survival analysis:
        - Higher score = higher predicted progression risk
        - The Breslow estimator (fit post-training) converts this to
          actual survival probabilities P(progression within X days)

    Architecture:
        LSTM_HIDDEN → 128 → ReLU → Dropout(0.3) → 1
    Converts fused MRI + clinical representation
    into Cox survival risk score.
    """

    def __init__(
        self,
        input_dim=384,
        hidden_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.head(x)
    
class GliomaModel(nn.Module):  
        """
        Multimodal spatiotemporal survival model

        Pipeline:
            MRI -> MobileNet -> LSTM
            Clinical -> MLP
            Fusion -> Prediction Head
            Output -> Cox log - risk score
        """ 

        def __init__(
                self,
                num_clinical_features,
                cnn_out_dim = 512,
                lstm_hidden =256,
                clinical_hidden=128,
                dropout = 0.3,
        ):
            super().__init__()


            # MRI  spatial encoder
            self.cnn_encoder = MobileNetEncoder(
                in_channels=8,
                out_dim=cnn_out_dim,
            )

            # temporal modeling
            self.temporal_model = nn.LSTM(
                input_size=cnn_out_dim,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,

            )

            # clinical feature encoder
            self.clinical_encoder = ClinicalEncoder(
                input_dim=num_clinical_features,
                hidden_dim=clinical_hidden,
                output_dim=clinical_hidden,
                dropout=dropout,
            )

            # fusion prediction head
            fusion_dim = lstm_hidden + clinical_hidden
            
            self.prediction_head = PredictionHead(
                input_dim=fusion_dim,
                hidden_dim=128,
                dropout=dropout,
            )

        def forward(self, x, seq_mask, clinical_feat):
            """
            Forward pass.

            Args:
                x:
                    MRI tensor
                    Shape: (B, T, C, D, H, W)

                seq_mask:
                    Sequence validity mask
                    Shape: (B, T)

                clinical_feat:
                    Clinical feature tensor
                    Shape: (B, num_clinical_features)

            Returns:
                risk:
                    Cox log-risk score
                    Shape: (B, 1)
            """
            B, T, C, D, H, W = x.shape

            # ------------------------------------------------------------
            # Reshape for slice-wise CNN processing
            # ------------------------------------------------------------
            x = x.view(B * T, C, D, H, W)

            slice_embeddings = []

            for d in range(D):
                slices = x[:, :, d, :, :]
                emb = self.cnn_encoder(slices)
                slice_embeddings.append(emb)

            # ------------------------------------------------------------
            # Aggregate slice embeddings
            # ------------------------------------------------------------
            slice_embeddings = torch.stack(slice_embeddings, dim=1)

            # Mean pooling across slices
            timepoint_embeddings = slice_embeddings.mean(dim=1)

            # Restore temporal dimension
            timepoint_embeddings = timepoint_embeddings.view(B, T, -1)

            # ------------------------------------------------------------
            # LSTM temporal modeling
            # ------------------------------------------------------------
            lstm_out, (h_n, c_n) = self.temporal_model(timepoint_embeddings)

            # Final hidden state
            mri_embedding = h_n[-1]

            # ------------------------------------------------------------
            # Clinical encoding
            # ------------------------------------------------------------
            clinical_embedding = self.clinical_encoder(clinical_feat)

            # ------------------------------------------------------------
            # Multimodal fusion
            # ------------------------------------------------------------
            fusion = torch.cat(
                [mri_embedding, clinical_embedding],
                dim=1,
            )

            # ------------------------------------------------------------
            # Survival prediction
            # ------------------------------------------------------------
            risk = self.prediction_head(fusion)

            return risk



# ===========================================================================
# § 8 — COX PROPORTIONAL HAZARDS LOSS
# ===========================================================================

def cox_ph_loss(
    risk:  torch.Tensor,
    time:  torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """
    Negative partial log-likelihood of the Cox proportional hazards model.

    What this loss measures:
        For each patient i who experienced progression (event=1), the Cox loss
        asks: "Among all patients still at risk at time t_i (those with
        t_j >= t_i), how well does the model rank this patient's risk score
        above the others?"

        A perfect model → high risk for patients who progressed early.
        A random model → loss ≈ log(risk_set_size) for each event.

    Mathematical formulation (for a batch):
        L = -Σ_{i: event_i=1} [ risk_i - log Σ_{j: t_j ≥ t_i} exp(risk_j) ]

    Implementation:
        Sort patients by descending time → the risk set for patient i becomes
        a prefix of the sorted array. Then logcumsumexp gives the log-sum
        efficiently for all events in one pass.

    Normalisation:
        Divide by number of events so loss scale is consistent across batches
        with different event rates.

    Important constraints:
        - Requires at least 2 events per batch (checked in train_one_epoch)
        - Returns 0.0 if no events (batch skipped in training)
        - Assumes proportional hazards: risk ratio between any two patients
          is constant over time

    Reference:
        Cox (1972), "Regression Models and Life-Tables." JRSS-B, 34(2):187-220.

    Args:
        risk  : (B,) — raw log-risk scores from PredictionHead (squeezed)
        time  : (B,) — observed times in days
        event : (B,) — event indicators: 1.0=progression, 0.0=censored

    Returns:
        Scalar tensor — mean negative partial log-likelihood over events
    """
    risk  = risk.float()
    time  = time.float()
    event = event.float()

    # Return 0 if no events — gradient would be undefined
    if event.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk.device)

    # Sort by descending time so risk set = prefix of sorted array
    order      = torch.argsort(time, descending=True)
    risk       = risk[order]
    event      = event[order]

    # logcumsumexp(risk)[i] = log Σ_{j ≤ i} exp(risk[j])
    # After descending sort: this equals log of risk set sum for each patient
    log_cumsum = torch.logcumsumexp(risk, dim=0)

    # Negative partial log-likelihood, summed over events only, then normalised
    loss = -torch.sum((risk - log_cumsum) * event)
    return loss / event.sum()


# ===========================================================================
# § 9 — PATIENT INDEX BUILDER (used by train.py)
# ===========================================================================

def build_patient_index(data_root: Path, patient_list: list) -> dict:
    """
    Build a mapping from patient ID to their sorted list of timepoint Paths.

    This index is consumed by GliomaDataset to locate MRI files on disk.
    It is separated from the Dataset class so the same index can be reused
    across multiple Dataset instantiations (e.g. across CV folds) without
    re-scanning the filesystem every time.

    Timepoint directories are sorted alphabetically:
        Timepoint_1 < Timepoint_2 < ... < Timepoint_6
    This preserves chronological order (assumes directory names sort correctly).

    Skips:
        - Patient directories that don't exist on disk (with a warning)
        - Patients with no timepoint subdirectories

    Args:
        data_root    : Root Path containing one subdirectory per patient
                       (e.g. Path("MU-Glioma-Post/"))
        patient_list : List of patient directory name strings to include
                       (e.g. ["PatientID_0001", "PatientID_0002", ...])

    Returns:
        dict mapping patient_id (str) → sorted list of timepoint Path objects
        Example:
            {
              "PatientID_0001": [
                  Path(".../PatientID_0001/Timepoint_1"),
                  Path(".../PatientID_0001/Timepoint_2"),
              ],
              ...
            }
    """
    index = {}

    for patient in patient_list:
        patient_path = data_root / patient

        if not patient_path.is_dir():
            continue   # Skip missing patient directories

        # List and sort timepoint subdirectories chronologically
        timepoints = sorted([
            tp for tp in os.listdir(patient_path)
            if (patient_path / tp).is_dir()
        ])

        if len(timepoints) == 0:
            continue   # Skip patients with no timepoint data

        index[patient] = [patient_path / tp for tp in timepoints]

    print(f"[INFO] Patient index built: {len(index)} patients indexed.")
    return index
