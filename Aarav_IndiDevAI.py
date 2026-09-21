# ============================================================
# IndiDevAI
# AI-Powered District Development Intelligence for India
# ============================================================
# IBM SkillsBuild Academic Internship 2026
# Data Analytics with AI
#
# Author  : Aarav
# Dataset : Primary Census Abstract (PCA), India & States/UTs
#           State and District Level, Census of India 2011
# Source  : Office of the Registrar General & Census Commissioner, India
#           https://censusindia.gov.in/nada/index.php/catalog/6191
# File    : data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx
# ============================================================


# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 2 — CONFIGURATION
# ============================================================

# Reproducibility seed used for all random-state parameters
RANDOM_SEED = 42

# Paths
DATA_PATH      = os.path.join("data", "raw", "DDW_PCA0000_2011_Indiastatedist.xlsx")
PROCESSED_DIR  = os.path.join("data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "district_analysis_ready.csv")

# Target geographic level and aggregation type
TARGET_LEVEL = "DISTRICT"
TARGET_TRU   = "Total"

# K-Means cluster count (to be tuned via elbow method in ML section)
N_CLUSTERS = 5

# Isolation Forest contamination proportion
CONTAMINATION = 0.05

# Streamlit page settings
PAGE_TITLE = "IndiDevAI — District Development Intelligence"
PAGE_ICON  = "🇮🇳"
LAYOUT     = "wide"

# Columns that must exist in the raw dataset for the pipeline to proceed
REQUIRED_STRUCTURAL_COLS = [
    "State", "District", "Level", "Name", "TRU",
    "TOT_P", "TOT_M", "TOT_F",
]

# All Census numeric columns that will be converted to float
CENSUS_NUMERIC_COLS = [
    "No_HH",
    "TOT_P", "TOT_M", "TOT_F",
    "P_06", "M_06", "F_06",
    "P_SC", "M_SC", "F_SC",
    "P_ST", "M_ST", "F_ST",
    "P_LIT", "M_LIT", "F_LIT",
    "P_ILL", "M_ILL", "F_ILL",
    "TOT_WORK_P", "TOT_WORK_M", "TOT_WORK_F",
    "MAINWORK_P", "MAINWORK_M", "MAINWORK_F",
    "MAIN_CL_P", "MAIN_CL_M", "MAIN_CL_F",
    "MAIN_AL_P", "MAIN_AL_M", "MAIN_AL_F",
    "MAIN_HH_P", "MAIN_HH_M", "MAIN_HH_F",
    "MAIN_OT_P", "MAIN_OT_M", "MAIN_OT_F",
    "MARGWORK_P", "MARGWORK_M", "MARGWORK_F",
    "MARG_CL_P", "MARG_CL_M", "MARG_CL_F",
    "MARG_AL_P", "MARG_AL_M", "MARG_AL_F",
    "MARG_HH_P", "MARG_HH_M", "MARG_HH_F",
    "MARG_OT_P", "MARG_OT_M", "MARG_OT_F",
    "NON_WORK_P", "NON_WORK_M", "NON_WORK_F",
]

# Derived feature names — used for validation and CSV export selection
DERIVED_FEATURE_COLS = [
    "Sex_Ratio",
    "Child_Pop_Pct",
    "SC_Pop_Pct",
    "ST_Pop_Pct",
    "Literacy_Rate",
    "Male_Literacy_Rate",
    "Female_Literacy_Rate",
    "Gender_Literacy_Gap",
    "Worker_Participation",
    "Female_Worker_Part",
    "Main_Worker_Pct",
    "Marginal_Worker_Pct",
    "Non_Worker_Pct",
    "Agri_Worker_Pct",
]


# ============================================================
# SECTION 3 — DATA LOADING
# ============================================================

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Census PCA Excel file from disk.

    Performs:
    - File-existence check before attempting to read.
    - Loads Sheet1 with all columns read as strings (dtype=str) to
      preserve zero-padded Census codes (e.g. State='01', District='001').
    - Validates that all required structural columns are present.
    - Returns the complete raw DataFrame without any filtering or
      type conversion so callers can inspect the full raw structure.

    Parameters
    ----------
    path : str
        Relative path to the .xlsx file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all rows and columns from Sheet1.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required structural columns are missing from the loaded sheet.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: '{path}'\n"
            "Expected location: data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx"
        )

    df = pd.read_excel(path, sheet_name="Sheet1", dtype=str)

    # Strip leading/trailing whitespace from all column names
    df.columns = df.columns.str.strip()

    # Verify required structural columns
    missing_cols = [c for c in REQUIRED_STRUCTURAL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns missing from dataset: {missing_cols}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    return df


# ============================================================
# SECTION 4 — DATA VALIDATION
# ============================================================

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run comprehensive data quality checks on the raw DataFrame.

    Checks performed:
    1.  Total row and column counts.
    2.  Unique values in 'Level' and 'TRU' (stripped).
    3.  Missing-value counts per column and total.
    4.  Fully duplicate row count.
    5.  Count of rows where Level == DISTRICT and TRU == Total.
    6.  Unique state codes found in the district-total subset.
    7.  Duplicate (State, District) pairs in the district-total subset
        — should be zero for a clean dataset.
    8.  Any (State, District) pairs with a missing State or District code.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by load_dataset().

    Returns
    -------
    dict
        Validation report with the following keys:
        total_rows, total_columns, unique_levels, unique_tru,
        missing_per_col, total_missing, duplicate_rows,
        district_total_rows, unique_states_in_district,
        duplicate_district_keys, missing_state_codes,
        missing_district_codes.
    """
    # Strip text columns for comparison (work on a view, not the original)
    level_series = df["Level"].str.strip() if "Level" in df.columns else pd.Series(dtype=str)
    tru_series   = df["TRU"].str.strip()   if "TRU"   in df.columns else pd.Series(dtype=str)

    district_mask = (level_series == TARGET_LEVEL) & (tru_series == TARGET_TRU)
    district_subset = df[district_mask]

    # Duplicate (State, District) keys within the district-total slice
    dup_keys = int(
        district_subset.duplicated(subset=["State", "District"]).sum()
        if all(c in district_subset.columns for c in ["State", "District"])
        else 0
    )

    report = {
        "total_rows":               len(df),
        "total_columns":            len(df.columns),
        "unique_levels":            sorted(df["Level"].dropna().str.strip().unique().tolist())
                                    if "Level" in df.columns else [],
        "unique_tru":               sorted(df["TRU"].dropna().str.strip().unique().tolist())
                                    if "TRU" in df.columns else [],
        "missing_per_col":          df.isnull().sum().to_dict(),
        "total_missing":            int(df.isnull().sum().sum()),
        "duplicate_rows":           int(df.duplicated().sum()),
        "district_total_rows":      int(district_mask.sum()),
        "unique_states_in_district": sorted(
            district_subset["State"].dropna().str.strip().unique().tolist()
        ) if "State" in district_subset.columns else [],
        "num_unique_states":        district_subset["State"].nunique()
                                    if "State" in district_subset.columns else 0,
        "duplicate_district_keys":  dup_keys,
        "missing_state_codes":      int(district_subset["State"].isnull().sum())
                                    if "State" in district_subset.columns else 0,
        "missing_district_codes":   int(district_subset["District"].isnull().sum())
                                    if "District" in district_subset.columns else 0,
    }
    return report


# ============================================================
# SECTION 5 — DATA CLEANING
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply justified cleaning operations to the raw DataFrame.

    Operations performed (in order):
    1.  Strip whitespace from all string columns (non-destructive
        normalisation — does not change values, only removes padding).
    2.  Drop columns that are entirely empty (all NaN). In the Census
        PCA file these are trailing placeholder columns CQ–CV.
    3.  Convert known Census numeric columns (CENSUS_NUMERIC_COLS) from
        string to float using pd.to_numeric(errors='coerce').
        Coercion produces NaN for cells that cannot be parsed; it does
        NOT replace NaN with zero.
    4.  Check for negative values in population and worker columns and
        report them — but do NOT remove or replace them automatically.
        Negative values are a data anomaly that must be investigated.

    What is intentionally NOT done here:
    - Missing values are NOT filled with zero. Census NaN values may
      represent genuinely zero populations (e.g. no ST population in a
      district) or legitimately absent data; zero-filling without
      investigation would distort the analysis.
    - No rows are dropped based on substantive criteria (those decisions
      happen in filter_district_data).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by load_dataset().

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input DataFrame.
    dict
        Cleaning report with keys: empty_cols_dropped, numeric_cols_converted,
        negative_value_counts.
    """
    df = df.copy()

    report = {}

    # 1. Strip whitespace from all object columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # 2. Drop entirely-empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    df.drop(columns=empty_cols, inplace=True)
    report["empty_cols_dropped"] = empty_cols

    # 3. Convert Census numeric columns to float (only those that exist)
    convertible = [c for c in CENSUS_NUMERIC_COLS if c in df.columns]
    df[convertible] = df[convertible].apply(pd.to_numeric, errors="coerce")
    report["numeric_cols_converted"] = convertible

    # 4. Check for negative values in population/worker columns — report only
    neg_counts = {}
    for col in convertible:
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            neg_counts[col] = n_neg
    report["negative_value_counts"] = neg_counts

    return df, report


# ============================================================
# SECTION 6 — DISTRICT FILTERING
# ============================================================

def filter_district_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the cleaned DataFrame to the primary analysis population:
    district-level Total records.

    Filter criteria:
        Level == 'DISTRICT'
        TRU   == 'Total'

    Post-filter steps:
    1.  Verify uniqueness of (State, District) pairs — duplicates are
        reported via a printed warning but are NOT silently dropped.
        (A duplicate would indicate a data anomaly that must be understood.)
    2.  Drop rows where TOT_P is zero or NaN. Districts with no reported
        population cannot contribute to rate calculations and are
        analytically unusable.
    3.  Reset index.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of clean_dataset).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame of district-level Total records.
    dict
        Filter report with keys: rows_before_filter, rows_after_filter,
        duplicate_keys_found, zero_pop_rows_dropped.
    """
    filter_report = {}
    filter_report["rows_before_filter"] = len(df)

    mask = (df["Level"].str.strip() == TARGET_LEVEL) & \
           (df["TRU"].str.strip()   == TARGET_TRU)
    district_df = df[mask].copy()

    # Check for duplicate (State, District) composite keys
    dup_mask = district_df.duplicated(subset=["State", "District"], keep=False)
    n_dups = int(dup_mask.sum())
    filter_report["duplicate_keys_found"] = n_dups
    if n_dups > 0:
        dup_records = district_df[dup_mask][["State", "District", "Name"]]
        print(
            f"[WARNING] {n_dups} rows share a duplicate (State, District) key "
            f"in the district-total slice:\n{dup_records.to_string(index=False)}"
        )

    # Drop zero/missing population rows
    zero_pop = district_df["TOT_P"].isna() | (district_df["TOT_P"] == 0)
    n_zero = int(zero_pop.sum())
    filter_report["zero_pop_rows_dropped"] = n_zero
    district_df = district_df[~zero_pop]

    district_df.reset_index(drop=True, inplace=True)
    filter_report["rows_after_filter"] = len(district_df)

    return district_df, filter_report


# ============================================================
# SECTION 7 — FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive analytical socioeconomic indicators from raw Census columns.

    Census population denominator note
    ------------------------------------
    The Census of India defines the effective literacy rate as literate
    persons as a percentage of the population aged 7 and above (children
    aged 0–6 are excluded because literacy is not enumerated for them).
    Therefore all literacy-rate denominators subtract the 0–6 age group
    (P_06 / M_06 / F_06) from the corresponding total population.

    Worker participation rate uses TOT_P as denominator, consistent with
    how the Census tabulates the overall work participation rate (total
    workers as a share of total population).

    Zero-denominator guard
    ----------------------
    All divisions use safe_div(), which returns NaN wherever the
    denominator is zero or NaN. NaN propagates naturally through
    pandas arithmetic so no silent distortion occurs.

    Derived features created
    ------------------------
    DEMOGRAPHIC
      Sex_Ratio          : (TOT_F / TOT_M) * 1000
      Child_Pop_Pct      : (P_06 / TOT_P) * 100
      SC_Pop_Pct         : (P_SC / TOT_P) * 100
      ST_Pop_Pct         : (P_ST / TOT_P) * 100

    EDUCATION
      Literacy_Rate      : (P_LIT / (TOT_P - P_06)) * 100
      Male_Literacy_Rate : (M_LIT / (TOT_M - M_06)) * 100
      Female_Literacy_Rate:(F_LIT / (TOT_F - F_06)) * 100
      Gender_Literacy_Gap: Male_Literacy_Rate - Female_Literacy_Rate

    EMPLOYMENT
      Worker_Participation  : (TOT_WORK_P / TOT_P) * 100
      Female_Worker_Part    : (TOT_WORK_F / TOT_F) * 100
      Main_Worker_Pct       : (MAINWORK_P / TOT_WORK_P) * 100
      Marginal_Worker_Pct   : (MARGWORK_P / TOT_WORK_P) * 100
      Non_Worker_Pct        : (NON_WORK_P / TOT_P) * 100
      Agri_Worker_Pct       : ((MAIN_CL_P + MAIN_AL_P) / TOT_WORK_P) * 100

    Parameters
    ----------
    df : pd.DataFrame
        Filtered district DataFrame (output of filter_district_data).

    Returns
    -------
    pd.DataFrame
        DataFrame with derived indicator columns appended.
    """
    df = df.copy()

    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        """Return num/den; NaN where den is zero or NaN."""
        den_safe = den.where(den > 0)          # zero → NaN
        return num / den_safe

    # ── DEMOGRAPHIC ──────────────────────────────────────────────
    if {"TOT_F", "TOT_M"}.issubset(df.columns):
        df["Sex_Ratio"] = safe_div(df["TOT_F"], df["TOT_M"]) * 1000

    if {"P_06", "TOT_P"}.issubset(df.columns):
        df["Child_Pop_Pct"] = safe_div(df["P_06"], df["TOT_P"]) * 100

    if {"P_SC", "TOT_P"}.issubset(df.columns):
        df["SC_Pop_Pct"] = safe_div(df["P_SC"], df["TOT_P"]) * 100

    if {"P_ST", "TOT_P"}.issubset(df.columns):
        df["ST_Pop_Pct"] = safe_div(df["P_ST"], df["TOT_P"]) * 100

    # ── EDUCATION ────────────────────────────────────────────────
    # Denominator = population aged 7+ (total minus children 0–6)
    if {"TOT_P", "P_06", "P_LIT"}.issubset(df.columns):
        pop_7plus = df["TOT_P"] - df["P_06"]
        df["Literacy_Rate"] = safe_div(df["P_LIT"], pop_7plus) * 100

    if {"TOT_M", "M_06", "M_LIT"}.issubset(df.columns):
        pop_m_7plus = df["TOT_M"] - df["M_06"]
        df["Male_Literacy_Rate"] = safe_div(df["M_LIT"], pop_m_7plus) * 100

    if {"TOT_F", "F_06", "F_LIT"}.issubset(df.columns):
        pop_f_7plus = df["TOT_F"] - df["F_06"]
        df["Female_Literacy_Rate"] = safe_div(df["F_LIT"], pop_f_7plus) * 100

    if {"Male_Literacy_Rate", "Female_Literacy_Rate"}.issubset(df.columns):
        df["Gender_Literacy_Gap"] = df["Male_Literacy_Rate"] - df["Female_Literacy_Rate"]

    # ── EMPLOYMENT ───────────────────────────────────────────────
    if {"TOT_WORK_P", "TOT_P"}.issubset(df.columns):
        df["Worker_Participation"] = safe_div(df["TOT_WORK_P"], df["TOT_P"]) * 100

    if {"TOT_WORK_F", "TOT_F"}.issubset(df.columns):
        df["Female_Worker_Part"] = safe_div(df["TOT_WORK_F"], df["TOT_F"]) * 100

    if {"MAINWORK_P", "TOT_WORK_P"}.issubset(df.columns):
        df["Main_Worker_Pct"] = safe_div(df["MAINWORK_P"], df["TOT_WORK_P"]) * 100

    if {"MARGWORK_P", "TOT_WORK_P"}.issubset(df.columns):
        df["Marginal_Worker_Pct"] = safe_div(df["MARGWORK_P"], df["TOT_WORK_P"]) * 100

    if {"NON_WORK_P", "TOT_P"}.issubset(df.columns):
        df["Non_Worker_Pct"] = safe_div(df["NON_WORK_P"], df["TOT_P"]) * 100

    if {"MAIN_CL_P", "MAIN_AL_P", "TOT_WORK_P"}.issubset(df.columns):
        agri = df["MAIN_CL_P"] + df["MAIN_AL_P"]
        df["Agri_Worker_Pct"] = safe_div(agri, df["TOT_WORK_P"]) * 100

    return df


# ============================================================
# SECTION 8 — FEATURE VALIDATION
# ============================================================

def validate_features(df: pd.DataFrame) -> dict:
    """
    Validate all derived indicator columns for data quality.

    For percentage / rate features (all except Sex_Ratio):
    - Count NaN values.
    - Count infinite values.
    - Count values below 0 (invalid).
    - Count values above 100 (invalid for a percentage).

    For Sex_Ratio:
    - Count NaN values.
    - Count infinite values.
    - Count values <= 0 (invalid).
    - Count values > 2000 (implausibly high; Census range is ~500–1500).

    Invalid values are REPORTED here but NOT automatically corrected.
    The caller must decide on the appropriate remediation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered district DataFrame.

    Returns
    -------
    dict
        Nested validation report. Top-level keys are feature names.
        Each value is a dict with keys: nan_count, inf_count,
        below_zero, above_100 (or above_2000 for Sex_Ratio).
    """
    report = {}

    pct_features = [f for f in DERIVED_FEATURE_COLS if f != "Sex_Ratio"]

    for feat in pct_features:
        if feat not in df.columns:
            report[feat] = {"status": "not_created"}
            continue

        series = df[feat]
        report[feat] = {
            "nan_count":  int(series.isna().sum()),
            "inf_count":  int(np.isinf(series.replace([np.inf, -np.inf], np.nan).fillna(0)
                                        .where(np.isinf(series), other=0)).sum()),
            "below_zero": int((series < 0).sum()),
            "above_100":  int((series > 100).sum()),
        }
        # Correct inf count (pandas isinf works on finite series)
        report[feat]["inf_count"] = int(np.isinf(series.dropna()).sum())

    if "Sex_Ratio" in df.columns:
        sr = df["Sex_Ratio"]
        report["Sex_Ratio"] = {
            "nan_count":     int(sr.isna().sum()),
            "inf_count":     int(np.isinf(sr.dropna()).sum()),
            "below_or_zero": int((sr <= 0).sum()),
            "above_2000":    int((sr > 2000).sum()),
        }
    else:
        report["Sex_Ratio"] = {"status": "not_created"}

    return report


# ============================================================
# SECTION 9 — SAVE PROCESSED DATA
# ============================================================

def save_processed_data(df: pd.DataFrame,
                        path: str = PROCESSED_PATH) -> None:
    """
    Save the analysis-ready district DataFrame to CSV.

    Columns saved include identification fields, all raw Census
    numeric columns present in the DataFrame, and all derived
    indicator columns that were successfully created.

    The raw Excel file is never modified by this function.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered district DataFrame.
    path : str
        Destination path for the CSV output.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Identification columns
    id_cols = ["State", "District", "Subdistt", "Level", "Name", "TRU"]
    id_cols_present = [c for c in id_cols if c in df.columns]

    # Raw Census numeric columns present
    census_present = [c for c in CENSUS_NUMERIC_COLS if c in df.columns]

    # Derived features present
    derived_present = [c for c in DERIVED_FEATURE_COLS if c in df.columns]

    # Build ordered column list, deduplicating while preserving order
    seen = set()
    export_cols = []
    for c in id_cols_present + census_present + derived_present:
        if c not in seen:
            export_cols.append(c)
            seen.add(c)

    df[export_cols].to_csv(path, index=False)
    print(f"[INFO] Processed dataset saved: {path} ({len(df)} rows, {len(export_cols)} columns)")


# ============================================================
# SECTION 10 — PIPELINE RUNNER (non-Streamlit)
# ============================================================

def run_pipeline(verbose: bool = True) -> dict:
    """
    Execute the full data pipeline from raw load to processed CSV.

    Steps:
    1. load_dataset       — load raw Excel
    2. validate_dataset   — raw quality checks
    3. clean_dataset      — type conversion, empty-column removal
    4. filter_district_data — apply Level/TRU filter
    5. engineer_features  — derive indicators
    6. validate_features  — check indicator ranges
    7. save_processed_data — write CSV to data/processed/

    Parameters
    ----------
    verbose : bool
        If True, print progress and summary to stdout.

    Returns
    -------
    dict
        Pipeline results with keys: raw_df, validation_report,
        clean_report, filter_report, district_df, featured_df,
        feature_validation_report.
    """
    if verbose:
        print("=" * 60)
        print("IndiDevAI — Data Pipeline")
        print("=" * 60)

    # Step 1 — Load
    if verbose:
        print("\n[1/7] Loading raw dataset...")
    raw_df = load_dataset()
    if verbose:
        print(f"      Loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    # Step 2 — Validate raw
    if verbose:
        print("\n[2/7] Validating raw dataset...")
    val_report = validate_dataset(raw_df)
    if verbose:
        print(f"      Unique Level values : {val_report['unique_levels']}")
        print(f"      Unique TRU values   : {val_report['unique_tru']}")
        print(f"      District-Total rows : {val_report['district_total_rows']}")
        print(f"      Duplicate rows      : {val_report['duplicate_rows']}")
        print(f"      Total missing cells : {val_report['total_missing']}")
        print(f"      Num unique states   : {val_report['num_unique_states']}")
        print(f"      Duplicate dist keys : {val_report['duplicate_district_keys']}")

    # Step 3 — Clean
    if verbose:
        print("\n[3/7] Cleaning dataset...")
    clean_df, clean_report = clean_dataset(raw_df)
    if verbose:
        print(f"      Empty columns dropped      : {clean_report['empty_cols_dropped']}")
        print(f"      Numeric columns converted  : {len(clean_report['numeric_cols_converted'])}")
        if clean_report["negative_value_counts"]:
            print(f"      [WARNING] Negative values found: {clean_report['negative_value_counts']}")
        else:
            print("      No negative values found in census numeric columns.")

    # Step 4 — Filter
    if verbose:
        print("\n[4/7] Filtering to district-level Total records...")
    district_df, filter_report = filter_district_data(clean_df)
    if verbose:
        print(f"      Rows before filter         : {filter_report['rows_before_filter']}")
        print(f"      Duplicate (State,Dist) keys: {filter_report['duplicate_keys_found']}")
        print(f"      Zero-population rows dropped:{filter_report['zero_pop_rows_dropped']}")
        print(f"      Final district records      : {filter_report['rows_after_filter']}")
        print(f"      Unique states/UTs           : {district_df['State'].nunique()}")

    # Step 5 — Feature engineering
    if verbose:
        print("\n[5/7] Engineering derived features...")
    featured_df = engineer_features(district_df)
    created = [c for c in DERIVED_FEATURE_COLS if c in featured_df.columns]
    if verbose:
        print(f"      Derived features created    : {len(created)}")
        for f in created:
            print(f"        - {f}")

    # Step 6 — Feature validation
    if verbose:
        print("\n[6/7] Validating derived features...")
    feat_val = validate_features(featured_df)
    issues_found = False
    for feat, result in feat_val.items():
        if result.get("status") == "not_created":
            continue
        nan_c   = result.get("nan_count", 0)
        inf_c   = result.get("inf_count", 0)
        neg_c   = result.get("below_zero", result.get("below_or_zero", 0))
        over_c  = result.get("above_100", result.get("above_2000", 0))
        if any([nan_c, inf_c, neg_c, over_c]):
            issues_found = True
            if verbose:
                print(f"      [CHECK] {feat}: NaN={nan_c}, Inf={inf_c}, "
                      f"<0={neg_c}, >limit={over_c}")
    if not issues_found and verbose:
        print("      All derived features passed range validation.")

    # Step 7 — Save
    if verbose:
        print("\n[7/7] Saving processed dataset...")
    save_processed_data(featured_df)

    if verbose:
        print("\n" + "=" * 60)
        print("Pipeline completed successfully.")
        print(f"Processed dataset: {PROCESSED_PATH}")
        print("=" * 60)

    return {
        "raw_df":                   raw_df,
        "validation_report":        val_report,
        "clean_report":             clean_report,
        "filter_report":            filter_report,
        "district_df":              district_df,
        "featured_df":              featured_df,
        "feature_validation_report": feat_val,
    }


# ============================================================
# SECTION 11 — EXPLORATORY DATA ANALYSIS (placeholder)
# ============================================================
# To be implemented in the EDA development phase.
# Planned analyses:
#   - Distribution of key indicators (histograms, box plots)
#   - State-wise aggregations and comparisons
#   - Top/bottom districts by indicator
#   - Correlation matrix of derived indicators
#   - Regional pattern identification

def run_eda(df: pd.DataFrame) -> dict:
    """Placeholder for EDA. To be implemented in EDA phase."""
    # TODO: implement EDA in development phase
    return {}


# ============================================================
# SECTION 12 — STATISTICAL ANALYSIS (placeholder)
# ============================================================
# Planned:
#   - Descriptive statistics (mean, median, std, IQR)
#   - Correlation analysis
#   - Regional and state-level aggregates

def run_statistical_analysis(df: pd.DataFrame) -> dict:
    """Placeholder for Statistical Analysis. To be implemented."""
    # TODO: implement statistical analysis in development phase
    return {}


# ============================================================
# SECTION 13 — MACHINE LEARNING: K-MEANS CLUSTERING (placeholder)
# ============================================================

def run_kmeans(df: pd.DataFrame, features: list, n_clusters: int = N_CLUSTERS):
    """Placeholder for K-Means clustering. To be implemented in ML phase."""
    # TODO: implement K-Means in ML development phase
    return df


# ============================================================
# SECTION 14 — PCA VISUALISATION (placeholder)
# ============================================================

def run_pca_analysis(df: pd.DataFrame, features: list, n_components: int = 2):
    """Placeholder for PCA. To be implemented in ML phase."""
    # TODO: implement PCA in ML development phase
    return df


# ============================================================
# SECTION 15 — ANOMALY DETECTION (placeholder)
# ============================================================

def run_anomaly_detection(df: pd.DataFrame, features: list,
                          contamination: float = CONTAMINATION):
    """Placeholder for Isolation Forest. To be implemented."""
    # TODO: implement Isolation Forest in anomaly detection phase
    return df


# ============================================================
# SECTION 16 — AI-ASSISTED INSIGHTS (placeholder)
# ============================================================
# Framework: Observations → Insights → Hypotheses → Recommendations
#
# Language guidelines (per project specification):
#   ✓  "is associated with"
#   ✓  "may indicate"
#   ✓  "suggests a possible relationship"
#   ✓  "shows a pattern"
#   ✗  "causes" / "leads to" / "because of"

def generate_insights(df: pd.DataFrame) -> list:
    """Placeholder for AI-assisted insights. To be implemented."""
    # TODO: implement insights generation in insights development phase
    return []


# ============================================================
# SECTION 17 — STREAMLIT DASHBOARD
# ============================================================

def page_home():
    """Render the Home / Project Overview page."""
    st.title("🇮🇳 IndiDevAI")
    st.subheader("AI-Powered District Development Intelligence for India")
    st.markdown("""
    **IndiDevAI** analyses Indian district-level demographic, education, and employment
    characteristics using the **Census of India 2011** — Primary Census Abstract (PCA).

    This application is developed as part of the **IBM SkillsBuild Academic Internship 2026**
    (Data Analytics with AI).

    ---
    ### What this application covers
    | Section | Description |
    |---------|-------------|
    | Dataset Overview | Raw dataset structure, dimensions, quality checks |
    | Demographics | Population, sex ratio, SC/ST distribution |
    | Education | Literacy rates, gender literacy gap |
    | Employment | Worker participation, agriculture vs other workers |
    | District Comparison | Side-by-side district profiles |
    | Exploratory Analysis | Distributions, correlations, state aggregates |
    | District Clustering | K-Means socioeconomic groupings |
    | PCA Visualisation | Dimensionality reduction and district profiles |
    | Anomaly Detection | Districts with unusual socioeconomic patterns |
    | AI-Assisted Insights | Observations → Insights → Hypotheses → Recommendations |
    | Recommendations | Data-driven policy directions |
    | Methodology | Project approach and technical notes |

    ---
    > **Note:** Census 2011 is a historical snapshot. This application focuses on pattern
    > discovery and profiling — not future predictions.
    """)


def page_dataset_overview(raw_df: pd.DataFrame, district_df: pd.DataFrame,
                          val_report: dict, clean_report: dict,
                          filter_report: dict, feat_val: dict):
    """Render the Dataset Overview page with full validation results."""
    st.title("📂 Dataset Overview")
    st.markdown("""
    **Source:** Primary Census Abstract (PCA), Census of India 2011  
    **Publisher:** Office of the Registrar General & Census Commissioner, India  
    **URL:** https://censusindia.gov.in/nada/index.php/catalog/6191
    """)

    # ── Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows",           f"{val_report['total_rows']:,}")
    c2.metric("Raw Columns",        f"{val_report['total_columns']}")
    c3.metric("District-Total Rows",f"{val_report['district_total_rows']:,}")
    c4.metric("States / UTs",       f"{val_report['num_unique_states']}")

    # ── Pipeline quality summary
    st.subheader("Pipeline Quality Summary")
    summary_data = {
        "Check": [
            "Duplicate rows (raw)",
            "Duplicate (State, District) keys",
            "Zero-population districts dropped",
            "Empty columns removed",
            "Negative census values",
            "Missing state codes (district slice)",
            "Missing district codes (district slice)",
        ],
        "Result": [
            val_report["duplicate_rows"],
            filter_report["duplicate_keys_found"],
            filter_report["zero_pop_rows_dropped"],
            len(clean_report["empty_cols_dropped"]),
            sum(clean_report["negative_value_counts"].values()),
            val_report["missing_state_codes"],
            val_report["missing_district_codes"],
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # ── Unique Level and TRU values
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Level Values")
        st.write(val_report["unique_levels"])
    with col_b:
        st.subheader("TRU Values")
        st.write(val_report["unique_tru"])

    # ── Feature validation
    st.subheader("Derived Feature Validation")
    fv_rows = []
    for feat, result in feat_val.items():
        if result.get("status") == "not_created":
            fv_rows.append({"Feature": feat, "Status": "not created",
                            "NaN": "-", "Inf": "-", "Out-of-range": "-"})
        else:
            nan_c  = result.get("nan_count", 0)
            inf_c  = result.get("inf_count", 0)
            oor    = result.get("below_zero", result.get("below_or_zero", 0)) + \
                     result.get("above_100",  result.get("above_2000", 0))
            status = "✅ OK" if (nan_c + inf_c + oor) == 0 else "⚠️ Issues"
            fv_rows.append({"Feature": feat, "Status": status,
                            "NaN": nan_c, "Inf": inf_c, "Out-of-range": oor})
    st.dataframe(pd.DataFrame(fv_rows), use_container_width=True, hide_index=True)

    # ── District sample
    st.subheader("District Dataset Sample (first 20 rows)")
    id_cols  = ["State", "District", "Name"]
    show_raw = [c for c in id_cols + ["TOT_P", "TOT_M", "TOT_F"] if c in district_df.columns]
    derived_present = [c for c in DERIVED_FEATURE_COLS if c in district_df.columns]
    st.dataframe(district_df[show_raw + derived_present].head(20), use_container_width=True)

    # ── Missing values in the raw dataset
    st.subheader("Missing Values — Raw Dataset (columns with any missing)")
    missing = pd.Series(val_report["missing_per_col"]).sort_values(ascending=False)
    missing_nonzero = missing[missing > 0]
    if missing_nonzero.empty:
        st.success("No missing values in any column of the raw dataset.")
    else:
        st.dataframe(
            missing_nonzero.reset_index().rename(
                columns={"index": "Column", 0: "Missing Count"}
            ),
            use_container_width=True,
            hide_index=True,
        )


def page_coming_soon(section_name: str):
    """Placeholder page for sections not yet implemented."""
    st.title(f"🚧 {section_name}")
    st.info(
        f"**{section_name}** will be implemented in the next development phase.  \n"
        "The data pipeline (loading, validation, cleaning, feature engineering) "
        "is complete and operational."
    )


# ============================================================
# SECTION 18 — MAIN APPLICATION
# ============================================================

def main():
    """
    Streamlit application entry point.

    Runs the full data pipeline (cached), then renders the
    selected dashboard section.
    """
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
    )

    # ── Sidebar navigation ────────────────────────────────────
    st.sidebar.title("IndiDevAI")
    st.sidebar.caption("Census of India 2011 — District Analytics")

    sections = [
        "Home",
        "Dataset Overview",
        "Demographics",
        "Education",
        "Employment",
        "District Comparison",
        "Exploratory Analysis",
        "District Clustering",
        "PCA Visualisation",
        "Anomaly Detection",
        "AI-Assisted Insights",
        "Recommendations",
        "Methodology / About",
    ]
    selection = st.sidebar.radio("Navigate", sections)

    # ── Data pipeline (cached so it only runs once per session) ──
    @st.cache_data(show_spinner="Running data pipeline…")
    def cached_pipeline():
        raw_df   = load_dataset()
        val_rep  = validate_dataset(raw_df)
        clean_df, clean_rep = clean_dataset(raw_df)
        dist_df, filt_rep   = filter_district_data(clean_df)
        feat_df  = engineer_features(dist_df)
        feat_val = validate_features(feat_df)
        return raw_df, val_rep, clean_rep, filt_rep, dist_df, feat_df, feat_val

    try:
        (raw_df, val_report, clean_report,
         filter_report, district_df, featured_df, feat_val) = cached_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(f"Dataset structure error: {e}")
        st.stop()

    # ── Page routing ───────────────────────────────────────────
    if selection == "Home":
        page_home()
    elif selection == "Dataset Overview":
        page_dataset_overview(raw_df, featured_df, val_report,
                              clean_report, filter_report, feat_val)
    elif selection == "Demographics":
        page_coming_soon("Demographics")
    elif selection == "Education":
        page_coming_soon("Education")
    elif selection == "Employment":
        page_coming_soon("Employment")
    elif selection == "District Comparison":
        page_coming_soon("District Comparison")
    elif selection == "Exploratory Analysis":
        page_coming_soon("Exploratory Analysis")
    elif selection == "District Clustering":
        page_coming_soon("District Clustering")
    elif selection == "PCA Visualisation":
        page_coming_soon("PCA Visualisation")
    elif selection == "Anomaly Detection":
        page_coming_soon("Anomaly Detection")
    elif selection == "AI-Assisted Insights":
        page_coming_soon("AI-Assisted Insights")
    elif selection == "Recommendations":
        page_coming_soon("Recommendations")
    elif selection == "Methodology / About":
        page_coming_soon("Methodology / About")

    st.sidebar.markdown("---")
    st.sidebar.caption("IBM SkillsBuild Internship 2026")


# ============================================================
# SECTION 19 — ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # When run directly (python Aarav_IndiDevAI.py), execute the
    # data pipeline in CLI mode to verify all steps and generate
    # the processed CSV. Streamlit runs via `streamlit run`.
    run_pipeline(verbose=True)
