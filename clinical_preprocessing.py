"""
clinical_preprocessing.py
==========================
Loads and preprocesses raw clinical Excel data for the MU-Glioma-Post dataset.

Column schema is taken directly from the 74-column Excel sheet.

Outputs a clean CSV with:
  - Curated, encoded, imputed, and scaled clinical features
  - Survival labels:
        event  : 1 = first progression occurred, 0 = censored
        time   : days to first progression  (event=1)
                 OR last available MRI scan  (event=0, censored)

Usage:
    python clinical_preprocessing.py
    -- or import preprocess_clinical() from train.py --
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ===========================================================================
# CONFIGURATION
# ===========================================================================
# Paths — update these to match your local or Colab environment

CLINICAL_FILE = Path(r"C:\Users\sanat\Desktop\MLPR_Mobile_Small\clinical_file.xlsx")
SHEET_INDEX   = 1          # 0-based sheet index in the Excel workbook
OUTPUT_CSV    = Path(r"C:\Users\sanat\Desktop\MLPR_Mobile_Small\clinical_processed.csv")

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

# Primary patient identifier column in the Excel sheet
PATIENT_ID_COL = "Patient_ID"

# Survival label source columns
# PROGRESSION_COL : binary flag — 1 = patient experienced confirmed progression
# PROG_TIME_COL   : number of days from diagnosis to first progression event
PROGRESSION_COL = "Progression"
PROG_TIME_COL   = "Time to First Progression (Days)"

# Columns to exclude from the feature matrix entirely.
# These are either identifiers, survival labels, or known data-leakage sources
# (i.e. information that would only be available AFTER the prediction horizon).
COLS_TO_EXCLUDE = [
    PATIENT_ID_COL,
    PROGRESSION_COL,
    PROG_TIME_COL,
    "Number of days from Diagnosis to date of Further Progression",  # future leakage
    "Number of days from Diagnosis to death (Days)",                 # future leakage
    "Overall Survival (Death)",                                      # future leakage
    "Other mutations/alterations",                                   # unstructured free-text
]

# ---------------------------------------------------------------------------
# MRI timepoint day columns
# Used ONLY for deriving censoring time for patients without progression.
# A censored patient's last observation time = their last available MRI day.
# ---------------------------------------------------------------------------
MRI_DAY_COLS = [
    "Number of Days from Diagnosis to 1st MRI (Timepoint_1) ",
    "Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ",
    "Number of Days from Diagnosis to 3rd MRI (Timepoint_3) ",
    "Number of Days from Diagnosis to 4th MRI (Timepoint_4) ",
    "Number of Days from Diagnosis to 5th MRI (Timepoint_5) ",
    "Number of Days from Diagnosis to 6th MRI (Timepoint_6) ",
]

# ---------------------------------------------------------------------------
# Negative value handling strategy
#
# ABS_VALUE_COLS:
#   "Number of days from Diagnosis to First surgery or procedure" can be
#   negative when surgery (e.g. biopsy) occurred BEFORE the formal diagnosis
#   date. This is clinically plausible — the biopsy result IS what led to
#   the diagnosis. We keep these patients and take |value|, since the
#   magnitude (proximity to diagnosis) is the meaningful signal.
#
# SET_NEGATIVE_TO_NAN_COLS:
#   For columns where a negative value is almost certainly a data entry error
#   (no clinical interpretation). These are set to NaN and then median-imputed.
# ---------------------------------------------------------------------------
ABS_VALUE_COLS = [
    "Number of days from Diagnosis to First surgery or procedure ",
]

SET_NEGATIVE_TO_NAN_COLS: list[str] = [
    # Add column names here if needed, e.g.:
    # "Number of days from Diagnosis to Radiation Therapy Start date",
]

# ---------------------------------------------------------------------------
# Feature groups — three lists that together define the feature matrix
# ---------------------------------------------------------------------------

# Continuous numeric features (will be median-imputed then StandardScaled)
NUMERIC_FEATURES = [
    "Age at diagnosis",

    # Surgery timing — abs() applied in handle_negative_values()
    "Number of days from Diagnosis to First surgery or procedure ",

    # Radiation therapy fields (sentinel-0 filled in engineer_features)
    "RT_start_days",       # days from diagnosis to RT start
    "RT_end_days",         # days from diagnosis to RT end
    "RT_num_fractions",    # number of radiation fractions delivered

    # Chemotherapy fields (sentinel-0 filled)
    "chemo_start_days",    # days from diagnosis to chemo start
    "chemo_end_days",      # days from diagnosis to chemo end

    # Immunotherapy fields (sentinel-0 filled)
    "immuno_start_days",   # days from diagnosis to immunotherapy start

    # Derived MRI count feature
    "num_mri_timepoints",  # how many longitudinal scans this patient had (1–6)

    # Radiation dose (cleaned from "60 Gy" format in clean_dose_column)
    "Dose",

    # MRI scan timing — encodes follow-up density and treatment timeline
    "Number of Days from Diagnosis to 1st MRI (Timepoint_1) ",
    "Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ",
    "Number of Days from Diagnosis to 3rd MRI (Timepoint_3) ",
    "Number of Days from Diagnosis to 4th MRI (Timepoint_4) ",
    "Number of Days from Diagnosis to 5th MRI (Timepoint_5) ",
    "Number of Days from Diagnosis to 6th MRI (Timepoint_6) ",
]

# String/object categorical features (LabelEncoded → float, NaN preserved)
CATEGORICAL_FEATURES = [
    "Sex at Birth",
    "Primary Diagnosis",
    "Grade of Primary Brain Tumor",
    "Previous Brain Tumor",
    "H3-3A mutation",
    "EGFR amplification",
    "ATRX mutation",
    "MGMT methylation",
    "BRAF V600E mutation",
    "TERT promoter mutation",
    "Chromosome 7 gain and Chromosome 10 loss",
    "1p/19q",
    "Initial Chemo Therapy",
    "Radiation Therapy",
]

# Integer columns already encoded as 0/1 in the Excel file — no encoding needed.
# These include derived treatment flags and clinical binary indicators.
BINARY_INT_FEATURES = [
    # Derived treatment-received flags (created in engineer_features)
    "received_radiation",
    "received_chemo",
    "received_immunotherapy",

    # Direct binary columns from Excel
    "Stereotactic Biopsy before Surgical Resection",
    "Type of 1st Progression",
    "Second Progression/Recurrence",
    "Type of 2nd Progression",
    "Multiple surgeries",
    "Hospice",
    "Overall Survival (Death)",
]


# ===========================================================================
# STEP 1 — Load raw Excel file
# ===========================================================================

def load_raw_clinical(filepath: Path, sheet: int) -> pd.DataFrame:
    """
    Load the raw clinical Excel file into a DataFrame.

    Args:
        filepath : Path to the .xlsx clinical data file
        sheet    : 0-based sheet index to read (SHEET_INDEX = 1)

    Returns:
        Raw DataFrame with all original columns (74 columns for this dataset)
    """
    print(f"[INFO] Loading: {filepath}  (sheet index={sheet})")
    df = pd.read_excel(filepath, sheet_name=sheet)
    print(f"[INFO] Raw shape: {df.shape}")
    return df


# ===========================================================================
# STEP 2 — Construct survival labels (event + time)
# ===========================================================================

def build_survival_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the two survival analysis target columns:

        event : int (0 or 1)
            1 = patient experienced confirmed tumor progression
            0 = censored (no progression observed by end of follow-up)

        time  : float (days)
            For event=1 → days from diagnosis to first progression (PROG_TIME_COL)
            For event=0 → last available MRI scan day (censoring time)

    Fallback chain if PROG_TIME_COL is missing for an event=1 patient:
        → last MRI day  → 0.0  (with a warning printed)

    Negative times are clipped to 0.0.

    Args:
        df : Raw clinical DataFrame (must contain PROGRESSION_COL and PROG_TIME_COL)

    Returns:
        DataFrame with two new columns: 'event' and 'time'
    """
    df = df.copy()

    # Validate required columns exist
    for col in [PROGRESSION_COL, PROG_TIME_COL]:
        if col not in df.columns:
            raise KeyError(
                f"Required column not found: '{col}'\n"
                f"Available: {df.columns.tolist()}"
            )

    # event: coerce to int, treat NaN as censored (0)
    df["event"] = (
        pd.to_numeric(df[PROGRESSION_COL], errors="coerce")
        .fillna(0).astype(int)
    )

    def _resolve_time(row):
        """
        For each patient row, determine the appropriate survival time:
          - event=1 → use PROG_TIME_COL if available
          - event=0 or missing → use the latest non-null MRI day (censoring time)
          - If no MRI day available → fallback to 0.0
        """
        if row["event"] == 1:
            t = pd.to_numeric(row.get(PROG_TIME_COL, np.nan), errors="coerce")
            if not pd.isna(t):
                return float(t)

        # Censored or missing progression time → last available MRI day
        mri_times = [
            float(pd.to_numeric(row.get(c, np.nan), errors="coerce"))
            for c in MRI_DAY_COLS
            if not pd.isna(pd.to_numeric(row.get(c, np.nan), errors="coerce"))
        ]
        return max(mri_times) if mri_times else 0.0

    df["time"] = df.apply(_resolve_time, axis=1).clip(lower=0.0)

    print(
        f"[INFO] Survival labels:\n"
        f"       Progression (event=1) : {df['event'].sum()}\n"
        f"       Censored    (event=0) : {(df['event']==0).sum()}\n"
        f"       Median time           : {df['time'].median():.1f} days\n"
        f"       time=0 (fallback)     : {(df['time']==0.0).sum()} patient(s)\n"
    )
    return df


# ===========================================================================
# STEP 3 — Handle known negative values
# ===========================================================================

def handle_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column-specific strategies for clinically negative values.

    Strategy A — ABS_VALUE_COLS (take absolute value):
        Surgery timing can be negative when surgery preceded diagnosis.
        |value| preserves the clinically useful magnitude (proximity).

        Example:
            -5  →  5   (biopsy 5 days BEFORE formal diagnosis)
             0  →  0   (surgery on same day as diagnosis)
            14  →  14  (surgery 14 days AFTER diagnosis)

    Strategy B — SET_NEGATIVE_TO_NAN_COLS (set to NaN, then median-impute):
        For columns where a negative value has no clinical interpretation
        and is almost certainly a data entry error.

    Args:
        df : DataFrame after survival labels have been added

    Returns:
        DataFrame with negative values handled per-column strategy
    """
    df = df.copy()

    # Strategy A: absolute value
    for col in ABS_VALUE_COLS:
        if col not in df.columns:
            print(f"[WARN] ABS_VALUE_COLS: column not found — skipped: '{col}'")
            continue
        n_neg = (pd.to_numeric(df[col], errors="coerce") < 0).sum()
        if n_neg > 0:
            print(f"[INFO] '{col}': {n_neg} negative value(s) → taking absolute value")
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()

    # Strategy B: set to NaN
    for col in SET_NEGATIVE_TO_NAN_COLS:
        if col not in df.columns:
            print(f"[WARN] SET_NEGATIVE_TO_NAN_COLS: column not found — skipped: '{col}'")
            continue
        mask  = pd.to_numeric(df[col], errors="coerce") < 0
        n_neg = mask.sum()
        if n_neg > 0:
            print(f"[INFO] '{col}': {n_neg} negative value(s) → set to NaN (will be imputed)")
            df.loc[mask, col] = np.nan

    print()
    return df


# ===========================================================================
# STEP 4 — Encode string categorical features
# ===========================================================================

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode each column in CATEGORICAL_FEATURES from string → float.

    Process per column:
        1. Cast to string (handles mixed types)
        2. Replace "nan" strings with "__missing__" sentinel
        3. Fit LabelEncoder on all unique values
        4. Transform → integer codes stored as float
        5. Re-convert __missing__ codes back to NaN
           (so the imputer can handle them in Step 6)

    This preserves NaN structure — NaN is not treated as a valid category.

    Args:
        df : DataFrame with raw string categorical columns

    Returns:
        DataFrame with categoricals replaced by float-coded integers (NaN preserved)
    """
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            print(f"[WARN] Categorical column not found — skipped: '{col}'")
            continue

        le = LabelEncoder()

        df[col] = df[col].astype(str)
        df[col] = df[col].replace("nan", "__missing__")

        le.fit(df[col])
        df[col] = le.transform(df[col]).astype(float)

        # Restore __missing__ as NaN so downstream imputer handles it
        if "__missing__" in le.classes_:
            missing_code = le.transform(["__missing__"])[0]
            df.loc[df[col] == missing_code, col] = np.nan

        print(f"[INFO] Encoded '{col}'")

    print()
    return df


def clean_dose_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the numeric part from the Dose column which stores values like
    '60 Gy', '54Gy', or 'NA'.

    Uses regex to pull the first integer/decimal number found in the string.
    Non-numeric entries (e.g. 'NA', 'Unknown') become NaN.

    Args:
        df : DataFrame containing a 'Dose' column

    Returns:
        DataFrame with 'Dose' replaced by float values (NaN where missing)
    """
    if "Dose" not in df.columns:
        print("[WARN] 'Dose' column not found — skipping.")
        return df

    df = df.copy()

    df["Dose"] = pd.to_numeric(
        df["Dose"].astype(str).str.extract(r'(\d+\.?\d*)')[0],
        errors="coerce"
    )

    print("[INFO] Cleaned 'Dose' column → numeric values extracted.\n")
    return df


# ===========================================================================
# STEP 4b — Domain-aware feature engineering
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw therapy and MRI timing columns.

    Key principle: Missing ≠ unknown — in this dataset, a missing value for
    a therapy column means that therapy was NOT given. We encode this
    explicitly with binary "received_X" flags and sentinel-0 timing columns,
    rather than leaving NaN for the imputer.

    Derived columns created:

        Radiation Therapy:
            received_radiation   : 1 if patient received any radiation, else 0
            RT_start_days        : days from diagnosis to RT start (0 if not received)
            RT_end_days          : days from diagnosis to RT end   (0 if not received)
            RT_num_fractions     : number of RT fractions          (0 if not received)

        Chemotherapy:
            received_chemo       : 1 if patient received initial chemo, else 0
            chemo_start_days     : days from diagnosis to chemo start (0 if not received)
            chemo_end_days       : days from diagnosis to chemo end   (0 if not received)

        Immunotherapy:
            received_immunotherapy : 1 if patient received immunotherapy, else 0
            immuno_start_days      : days from diagnosis to immuno start (0 if not received)

        MRI Timing:
            num_mri_timepoints   : count of non-null MRI scan days (1–6)
            MRI_DAY_COLS         : missing scan slots filled with 0 as sentinel

    Sparse columns (< 10% non-null) are dropped to avoid noise.
    Original raw therapy columns are dropped after derivation.

    Args:
        df : DataFrame after categorical encoding

    Returns:
        DataFrame with new derived columns and cleaned-up raw columns removed
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. RADIATION THERAPY
    #    Any non-null value in the raw "Radiation Therapy" column
    #    (e.g. "Yes", "Proton Therapy") means the patient received radiation.
    # ------------------------------------------------------------------
    df["received_radiation"] = df["Radiation Therapy"].notna().astype(float)
    df["RT_start_days"]   = df["Number of days from Diagnosis to Radiation Therapy Start date"].fillna(0)
    df["RT_end_days"]     = df["Number of days from Diagnosis to Radiation Therapy end date"].fillna(0)
    df["RT_num_fractions"]= df["Number of Fractions"].fillna(0)

    # ------------------------------------------------------------------
    # 2. INITIAL CHEMOTHERAPY
    #    Only value observed is "Yes" — presence = received chemo.
    # ------------------------------------------------------------------
    df["received_chemo"]   = df["Initial Chemo Therapy"].notna().astype(float)
    df["chemo_start_days"] = df[" Number of days from Diagnosis to Initial Chemo Therapy Start date"].fillna(0)
    df["chemo_end_days"]   = df[" Number of days from Diagnosis to Initial Chemo Therapy end date"].fillna(0)

    # ------------------------------------------------------------------
    # 3. IMMUNOTHERAPY
    #    Values are drug names — any non-null = received immunotherapy.
    # ------------------------------------------------------------------
    df["received_immunotherapy"] = df["Immuno therapy"].notna().astype(float)
    df["immuno_start_days"]      = df["Number of Days from Diagnosis to Start Immunotherapy "].fillna(0)

    # ------------------------------------------------------------------
    # 4. MRI TIMEPOINT TIMING
    #    Count of actual scans per patient + fill missing slots with 0.
    #    0 is used as a sentinel meaning "scan slot not used", distinguishable
    #    from a real scan at day 0 only by the received_X flag.
    # ------------------------------------------------------------------
    df["num_mri_timepoints"] = df[MRI_DAY_COLS].notna().sum(axis=1).astype(float)
    for col in MRI_DAY_COLS:
        df[col] = df[col].fillna(0)

    # ------------------------------------------------------------------
    # 5. PREVIOUS BRAIN TUMOR sub-columns
    #    Only 11-13 of 203 patients have previous brain tumor data.
    #    These sub-columns are too sparse to be useful — drop them.
    # ------------------------------------------------------------------
    sparse_prev = [
        "Type of previous brain tumor",
        "Year of previous surgery",
        "Grade of Previous brain tumor",
    ]
    df.drop(columns=[c for c in sparse_prev if c in df.columns], inplace=True)

    # ------------------------------------------------------------------
    # 6. Drop raw therapy columns that have been replaced by derived cols above
    # ------------------------------------------------------------------
    raw_cols_to_drop = [
        "Radiation Therapy",
        "Number of days from Diagnosis to Radiation Therapy Start date",
        "Number of days from Diagnosis to Radiation Therapy end date",
        "Number of Fractions",
        "Initial Chemo Therapy",
        " Number of days from Diagnosis to Initial Chemo Therapy Start date",
        " Number of days from Diagnosis to Initial Chemo Therapy end date",
        "Immuno therapy",
        "Number of Days from Diagnosis to Start Immunotherapy ",
        # Brachytherapy — too rare/sparse to include
        "Brachy therapy",
        "Number of Days from Diagnosis to the day of Insertion of Brachytherapy ",
        # Additional / secondary therapy columns — sparse and heterogeneous
        "Additional Therapy",
        "Cycle length of Additional Therapy (q days)",
        "Number of Days from Diagnosis to Starting Additional Therapy ",
        "Number of Days from Diagnosis to Complete Additional Therapy ",
        "Number of Cycles of Additional Therapy",
        "2nd_Additional Therapy",
        "Cycle length of 2nd_Additional Therapy (q days)",
        "Number of Days from Diagnosis to Starting 2nd_Additional Therapy ",
        "Number of Days from Dagnosis to Complete 2nd_Additional Therapy ",
        "Number of Cycles of 2nd_Additional Therapy",
        "Other Types of Therapy (LITT, more chemo, proton therapy)",
        "Number of Days from Diagnosis to Start Other Additional Therapy ",
        "Number of Days from Diagnosis to Complete Other Additional Therapy ",
        "Treatment started after 2nd progression",
        "Days from Diagnosis to new treatment",
        "Cycle length of Immunotherapy (q days)",
        "Number of Days from Diagnosis to Complete Immunotherapy",
        "Number of Cycles of Immunotherapy",
    ]
    df.drop(
        columns=[c for c in raw_cols_to_drop if c in df.columns],
        inplace=True
    )

    print(
        "[INFO] Feature engineering complete — derived columns added:\n"
        "       received_radiation, RT_start_days, RT_end_days, RT_num_fractions\n"
        "       received_chemo, chemo_start_days, chemo_end_days\n"
        "       received_immunotherapy, immuno_start_days\n"
        "       num_mri_timepoints\n"
    )
    return df


# ===========================================================================
# STEP 5 — Validate and select feature columns
# ===========================================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Assemble the final feature column list from the three feature groups,
    then drop:
        - Columns not present in the DataFrame (with warnings)
        - Zero-variance (constant) columns — carry no information

    Note: columns with > 60% missing are NOT dropped here because
    engineer_features() already handles missingness via sentinel-0
    or explicit imputation strategy. The imputer in Step 6 is a
    last-resort fallback only.

    Args:
        df : Fully engineered DataFrame

    Returns:
        List of column names to use as the feature matrix
    """
    # Deduplicate while preserving order
    requested = list(dict.fromkeys(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_INT_FEATURES
    ))

    present, absent = [], []
    for col in requested:
        (present if col in df.columns else absent).append(col)
    if absent:
        print(f"[WARN] Columns not found in DataFrame:\n  {absent}\n")

    # Log any unexpected NaN that survived engineering
    miss_frac     = df[present].isnull().mean()
    still_missing = miss_frac[miss_frac > 0]
    if not still_missing.empty:
        print(
            f"[WARN] Unexpected missing values after engineering:\n"
            f"{still_missing.to_string()}\n"
            f"       These will be median-imputed as fallback."
        )

    keep = present

    # Drop zero-variance columns (constant values carry no signal)
    variances = df[keep].var(numeric_only=True)
    zero_var  = variances[variances <= 1e-8].index.tolist()
    keep      = [c for c in keep if c not in zero_var]
    if zero_var:
        print(f"[INFO] Dropped (zero variance): {zero_var}")

    print(f"[INFO] Features retained: {len(keep)} / {len(requested)}\n")
    return keep


# ===========================================================================
# STEP 6 — Impute missing values + apply StandardScaler
# ===========================================================================

def impute_and_scale(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Fill any remaining NaN values and scale all features to zero mean / unit variance.

    Imputation strategy by column type:

        Sentinel-0 columns  (MRI days, therapy timing):
            → Should already be 0 after engineer_features().
              Warns and force-fills if any NaN remains.

        Binary int columns  (0/1 flags):
            → Mode imputation (fills with most common value: 0 or 1).

        Categorical encoded columns:
            → Mode imputation (fills with most frequent encoded category).

        Remaining numeric columns:
            → Median imputation as last-resort fallback ONLY.
              These should not have NaN after engineering — warns if found.

    After imputation: StandardScaler applied to ALL feature columns.
        Each feature becomes: (x - mean) / std
        This prevents large-magnitude features (e.g. days: 0–1000) from
        dominating the model over small-magnitude features (e.g. binary: 0/1).

    Args:
        df           : Engineered DataFrame with potential NaN values
        feature_cols : List of column names to impute and scale

    Returns:
        DataFrame with all NaN filled and feature columns StandardScaled
    """
    df = df.copy()
    if not feature_cols:
        print("[WARN] No feature columns to process.")
        return df

    # --- Sentinel-0 columns: should already be clean ---
    sentinel_cols = (
        [c for c in MRI_DAY_COLS if c in feature_cols]
        + [c for c in [
            "RT_start_days", "RT_end_days", "RT_num_fractions",
            "chemo_start_days", "chemo_end_days", "immuno_start_days",
        ] if c in feature_cols]
    )
    still_null = df[sentinel_cols].isnull().sum()
    still_null = still_null[still_null > 0]
    if not still_null.empty:
        print(
            f"[WARN] Sentinel columns still have NaN — force-filling 0:\n"
            f"       {still_null.to_dict()}"
        )
        df[sentinel_cols] = df[sentinel_cols].fillna(0)

    # --- Binary int columns: mode imputation ---
    binary_cols = [c for c in BINARY_INT_FEATURES if c in feature_cols]
    for col in binary_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            print(f"[INFO] Mode-imputed binary column '{col}' → {mode_val}")

    # --- Categorical encoded columns: mode imputation ---
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col]  = df[col].fillna(mode_val)
            print(f"[INFO] Mode-imputed categorical column '{col}' → {mode_val}")

    # --- Remaining numeric: median fallback ---
    remaining_numeric = [
        c for c in feature_cols
        if c not in sentinel_cols + binary_cols + cat_cols
    ]
    for col in remaining_numeric:
        if df[col].isnull().any():
            n_null  = df[col].isnull().sum()
            med_val = df[col].median()
            df[col] = df[col].fillna(med_val)
            print(
                f"[WARN] Unexpected NaN in '{col}' ({n_null} rows) "
                f"— median fallback: {med_val:.1f}"
            )

    # --- StandardScaler: zero mean, unit variance ---
    scaler           = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"\n[INFO] StandardScaler applied → {len(feature_cols)} columns.")

    return df


# ===========================================================================
# STEP 7 — Dimensionality reduction via PCA
# ===========================================================================

def apply_pca(df: pd.DataFrame, feature_cols: list, variance_threshold: float = 0.95):
    """
    Reduce the feature matrix dimensionality using PCA.

    PCA projects the features onto orthogonal principal components ordered
    by explained variance. We retain enough components to explain at least
    `variance_threshold` (default 95%) of the total variance.

    Why PCA here?
        The clinical feature matrix has ~30+ columns, many correlated
        (e.g. RT_start_days and RT_end_days). PCA removes redundancy
        and reduces overfitting risk, while preserving most information.

    Args:
        df                 : Scaled DataFrame (output of impute_and_scale)
        feature_cols       : List of scaled feature column names to reduce
        variance_threshold : Float in (0, 1] — fraction of variance to retain

    Returns:
        df_pca   : DataFrame with PCA_1, PCA_2, ... columns (same row index)
        pca_cols : List of new PCA column names
    """
    df = df.copy()

    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(df[feature_cols])

    # Name new columns sequentially
    pca_cols = [f"PCA_{i+1}" for i in range(X_pca.shape[1])]
    df_pca   = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

    print(f"[INFO] PCA applied:")
    print(f"       Original features : {len(feature_cols)}")
    print(f"       Reduced features  : {len(pca_cols)}")
    print(f"       Variance retained : {sum(pca.explained_variance_ratio_):.4f}\n")

    return df_pca, pca_cols


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def preprocess_clinical(
    filepath:   Path = CLINICAL_FILE,
    sheet:       int = SHEET_INDEX,
    output_csv: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Execute the full clinical preprocessing pipeline in order:

        Step 1 : Load raw Excel file
        Step 2 : Build survival labels (event, time)
        Step 3 : Handle negative values (abs or NaN strategy)
        Step 4 : Clean Dose column + encode categoricals
        Step 4b: Engineer derived therapy/MRI features
        Step 5 : Validate and select feature columns
        Step 6 : Impute NaN + StandardScale all features
        Step 7 : Apply PCA (95% variance retention)
        Save   : Write final CSV to disk

    Output DataFrame schema:
        Patient_ID | PCA_1 | PCA_2 | ... | PCA_N | event | time

    Args:
        filepath   : Path to raw clinical .xlsx file
        sheet      : 0-based sheet index
        output_csv : Path where processed CSV will be saved

    Returns:
        Processed DataFrame ready to be consumed by GliomaDataset in train.py
    """
    df = load_raw_clinical(filepath, sheet)       # Step 1
    df = build_survival_labels(df)                # Step 2
    df = handle_negative_values(df)               # Step 3
    df = clean_dose_column(df)                    # Step 4
    df = encode_categoricals(df)                  # Step 4
    df = engineer_features(df)                    # Step 4b
    feature_cols = get_feature_columns(df)        # Step 5
    df = impute_and_scale(df, feature_cols)       # Step 6

    # Step 7: PCA dimensionality reduction
    df_pca, pca_cols = apply_pca(df, feature_cols)

    # Assemble final output: Patient_ID + event + time + PCA features
    df_out = pd.concat([
        df[[PATIENT_ID_COL, "event", "time"]],
        df_pca
    ], axis=1).reset_index(drop=True)

    # Save to disk
    os.makedirs(output_csv.parent, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print(f"[INFO] Saved → {output_csv}")
    print(
        f"[INFO] Final shape: {df_out.shape}  "
        f"({len(pca_cols)} PCA features + Patient_ID + event + time)\n"
    )

    return df_out


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    df = preprocess_clinical()
    print("[PREVIEW] First 5 rows:")
    print(df.head(5).to_string())
    print("\n[DTYPES]")
    print(df.dtypes.to_string())
