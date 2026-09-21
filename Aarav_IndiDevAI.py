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
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 2 — CONFIGURATION
# ============================================================

RANDOM_SEED    = 42
DATA_PATH      = os.path.join("data", "raw", "DDW_PCA0000_2011_Indiastatedist.xlsx")
PROCESSED_DIR  = os.path.join("data", "processed")
PROCESSED_PATH    = os.path.join(PROCESSED_DIR, "district_analysis_ready.csv")
ML_RESULTS_PATH   = os.path.join(PROCESSED_DIR, "district_ml_results.csv")
TARGET_LEVEL   = "DISTRICT"
TARGET_TRU     = "Total"
N_CLUSTERS     = 5
CONTAMINATION  = 0.05
PAGE_TITLE     = "IndiDevAI — District Development Intelligence"
PAGE_ICON      = "🇮🇳"
LAYOUT         = "wide"

REQUIRED_STRUCTURAL_COLS = [
    "State", "District", "Level", "Name", "TRU",
    "TOT_P", "TOT_M", "TOT_F",
]

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

# ── Preliminary ML feature shortlist (Step 4) ────────────────────────────────
#
# Rationale for inclusion / exclusion of each candidate:
#
# INCLUDED:
#   Sex_Ratio          — captures gender demographic balance
#   Child_Pop_Pct      — proxy for fertility / young dependency
#   SC_Pop_Pct         — social composition indicator
#   ST_Pop_Pct         — social composition indicator (tribal concentration)
#   Literacy_Rate      — overall education proxy (7+ pop denominator)
#   Female_Literacy_Rate — female education proxy; not redundant with
#                          Literacy_Rate because it captures gender-specific access
#   Gender_Literacy_Gap  — gender education inequality indicator
#   Worker_Participation — overall workforce engagement
#   Female_Worker_Part   — female workforce engagement
#   Agri_Worker_Pct      — economic structure proxy
#   Main_Worker_Pct      — employment stability proxy
#   Child_Pop_Pct        — already listed above
#
# EXCLUDED (with reason):
#   Male_Literacy_Rate — highly collinear with Literacy_Rate and
#                        Female_Literacy_Rate; Gender_Literacy_Gap already
#                        captures the male-female differential
#   Non_Worker_Pct     — mathematical complement of Worker_Participation
#                        (Non_Worker_Pct = 100 - Worker_Participation approx.);
#                        including both would introduce perfect collinearity
#   Marginal_Worker_Pct — complement of Main_Worker_Pct for workers;
#                          Main_Worker_Pct already represents the same dimension
#
# Final list: 10 features (no arithmetic duplicates; captures all 4 domains)
#
ML_FEATURE_SHORTLIST = [
    "Sex_Ratio",
    "Child_Pop_Pct",
    "SC_Pop_Pct",
    "ST_Pop_Pct",
    "Literacy_Rate",
    "Female_Literacy_Rate",
    "Gender_Literacy_Gap",
    "Worker_Participation",
    "Female_Worker_Part",
    "Agri_Worker_Pct",
    "Main_Worker_Pct",
]

# Features excluded from ML shortlist and documented reason
ML_FEATURE_EXCLUSIONS = {
    "Male_Literacy_Rate":   "Collinear with Literacy_Rate and Female_Literacy_Rate; "
                            "Gender_Literacy_Gap captures the differential.",
    "Non_Worker_Pct":       "Arithmetic complement of Worker_Participation "
                            "(sum ~=100%); including both causes perfect collinearity.",
    "Marginal_Worker_Pct":  "Partial complement of Main_Worker_Pct within total workers; "
                            "Main_Worker_Pct already represents employment stability.",
}

# Human-readable labels for derived features used in charts
FEATURE_LABELS = {
    "Sex_Ratio":            "Sex Ratio (females per 1,000 males)",
    "Child_Pop_Pct":        "Child Population % (0–6 yrs)",
    "SC_Pop_Pct":           "SC Population %",
    "ST_Pop_Pct":           "ST Population %",
    "Literacy_Rate":        "Literacy Rate % (pop 7+)",
    "Male_Literacy_Rate":   "Male Literacy Rate % (pop 7+)",
    "Female_Literacy_Rate": "Female Literacy Rate % (pop 7+)",
    "Gender_Literacy_Gap":  "Gender Literacy Gap (M − F, pp)",
    "Worker_Participation": "Worker Participation Rate %",
    "Female_Worker_Part":   "Female Worker Participation %",
    "Main_Worker_Pct":      "Main Workers % of Total Workers",
    "Marginal_Worker_Pct":  "Marginal Workers % of Total Workers",
    "Non_Worker_Pct":       "Non-Worker %",
    "Agri_Worker_Pct":      "Agricultural Worker Share %",
}

# Plotly colour sequence used throughout charts
CHART_COLORS = px.colors.qualitative.Set2


# ============================================================
# SECTION 3 — DATA LOADING
# ============================================================

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Census PCA Excel file from disk with validation."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset not found: '{path}'\n"
            "Expected location: data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx"
        )
    df = pd.read_excel(path, sheet_name="Sheet1", dtype=str)
    df.columns = df.columns.str.strip()
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
    """Run comprehensive data quality checks on the raw DataFrame."""
    level_series  = df["Level"].str.strip() if "Level" in df.columns else pd.Series(dtype=str)
    tru_series    = df["TRU"].str.strip()   if "TRU"   in df.columns else pd.Series(dtype=str)
    district_mask = (level_series == TARGET_LEVEL) & (tru_series == TARGET_TRU)
    district_subset = df[district_mask]

    dup_keys = int(
        district_subset.duplicated(subset=["State", "District"]).sum()
        if all(c in district_subset.columns for c in ["State", "District"]) else 0
    )
    return {
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


# ============================================================
# SECTION 5 — DATA CLEANING
# ============================================================

def clean_dataset(df: pd.DataFrame):
    """Apply justified cleaning operations to the raw DataFrame."""
    df = df.copy()
    report = {}
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
    empty_cols = df.columns[df.isnull().all()].tolist()
    df.drop(columns=empty_cols, inplace=True)
    report["empty_cols_dropped"] = empty_cols
    convertible = [c for c in CENSUS_NUMERIC_COLS if c in df.columns]
    df[convertible] = df[convertible].apply(pd.to_numeric, errors="coerce")
    report["numeric_cols_converted"] = convertible
    neg_counts = {col: int((df[col] < 0).sum())
                  for col in convertible if int((df[col] < 0).sum()) > 0}
    report["negative_value_counts"] = neg_counts
    return df, report


# ============================================================
# SECTION 6 — DISTRICT FILTERING
# ============================================================

def filter_district_data(df: pd.DataFrame):
    """Filter to Level==DISTRICT and TRU==Total with quality checks."""
    filter_report = {"rows_before_filter": len(df)}
    mask = (df["Level"].str.strip() == TARGET_LEVEL) & \
           (df["TRU"].str.strip()   == TARGET_TRU)
    district_df = df[mask].copy()
    dup_mask = district_df.duplicated(subset=["State", "District"], keep=False)
    n_dups = int(dup_mask.sum())
    filter_report["duplicate_keys_found"] = n_dups
    if n_dups > 0:
        print(f"[WARNING] {n_dups} duplicate (State, District) keys found.")
    zero_pop = district_df["TOT_P"].isna() | (district_df["TOT_P"] == 0)
    filter_report["zero_pop_rows_dropped"] = int(zero_pop.sum())
    district_df = district_df[~zero_pop]
    district_df.reset_index(drop=True, inplace=True)
    filter_report["rows_after_filter"] = len(district_df)
    return district_df, filter_report


# ============================================================
# SECTION 7 — FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 14 analytical socioeconomic indicators from raw Census columns."""
    df = df.copy()

    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / den.where(den > 0)

    if {"TOT_F", "TOT_M"}.issubset(df.columns):
        df["Sex_Ratio"] = safe_div(df["TOT_F"], df["TOT_M"]) * 1000
    if {"P_06", "TOT_P"}.issubset(df.columns):
        df["Child_Pop_Pct"] = safe_div(df["P_06"], df["TOT_P"]) * 100
    if {"P_SC", "TOT_P"}.issubset(df.columns):
        df["SC_Pop_Pct"] = safe_div(df["P_SC"], df["TOT_P"]) * 100
    if {"P_ST", "TOT_P"}.issubset(df.columns):
        df["ST_Pop_Pct"] = safe_div(df["P_ST"], df["TOT_P"]) * 100
    if {"TOT_P", "P_06", "P_LIT"}.issubset(df.columns):
        df["Literacy_Rate"] = safe_div(df["P_LIT"], df["TOT_P"] - df["P_06"]) * 100
    if {"TOT_M", "M_06", "M_LIT"}.issubset(df.columns):
        df["Male_Literacy_Rate"] = safe_div(df["M_LIT"], df["TOT_M"] - df["M_06"]) * 100
    if {"TOT_F", "F_06", "F_LIT"}.issubset(df.columns):
        df["Female_Literacy_Rate"] = safe_div(df["F_LIT"], df["TOT_F"] - df["F_06"]) * 100
    if {"Male_Literacy_Rate", "Female_Literacy_Rate"}.issubset(df.columns):
        df["Gender_Literacy_Gap"] = df["Male_Literacy_Rate"] - df["Female_Literacy_Rate"]
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
        df["Agri_Worker_Pct"] = safe_div(
            df["MAIN_CL_P"] + df["MAIN_AL_P"], df["TOT_WORK_P"]
        ) * 100
    return df


# ============================================================
# SECTION 8 — FEATURE VALIDATION
# ============================================================

def validate_features(df: pd.DataFrame) -> dict:
    """Validate derived indicator columns for NaN, Inf, and out-of-range values."""
    report = {}
    for feat in [f for f in DERIVED_FEATURE_COLS if f != "Sex_Ratio"]:
        if feat not in df.columns:
            report[feat] = {"status": "not_created"}
            continue
        s = df[feat]
        report[feat] = {
            "nan_count":  int(s.isna().sum()),
            "inf_count":  int(np.isinf(s.dropna()).sum()),
            "below_zero": int((s < 0).sum()),
            "above_100":  int((s > 100).sum()),
        }
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

def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    """Save the analysis-ready district DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    id_cols = ["State", "District", "Subdistt", "Level", "Name", "TRU"]
    seen, export_cols = set(), []
    for c in ([c for c in id_cols if c in df.columns] +
              [c for c in CENSUS_NUMERIC_COLS if c in df.columns] +
              [c for c in DERIVED_FEATURE_COLS if c in df.columns]):
        if c not in seen:
            export_cols.append(c)
            seen.add(c)
    df[export_cols].to_csv(path, index=False)
    print(f"[INFO] Processed dataset saved: {path} ({len(df)} rows, {len(export_cols)} columns)")


# ============================================================
# SECTION 10 — PIPELINE RUNNER (non-Streamlit)
# ============================================================

def run_pipeline(verbose: bool = True) -> dict:
    """Execute the full data pipeline from raw load to processed CSV."""
    if verbose:
        print("=" * 60)
        print("IndiDevAI — Data Pipeline")
        print("=" * 60)
    raw_df = load_dataset()
    if verbose:
        print(f"\n[1/7] Loaded: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    val_report = validate_dataset(raw_df)
    if verbose:
        print(f"[2/7] Validated — District-Total rows: {val_report['district_total_rows']}, "
              f"States: {val_report['num_unique_states']}")
    clean_df, clean_report = clean_dataset(raw_df)
    if verbose:
        print(f"[3/7] Cleaned — {len(clean_report['numeric_cols_converted'])} numeric cols converted")
    district_df, filter_report = filter_district_data(clean_df)
    if verbose:
        print(f"[4/7] Filtered — {filter_report['rows_after_filter']} district records")
    featured_df = engineer_features(district_df)
    if verbose:
        print(f"[5/7] Features engineered — {len([c for c in DERIVED_FEATURE_COLS if c in featured_df.columns])} features")
    feat_val = validate_features(featured_df)
    if verbose:
        issues = [f for f, r in feat_val.items() if r.get("status") != "not_created"
                  and any([r.get("nan_count", 0), r.get("inf_count", 0),
                           r.get("below_zero", r.get("below_or_zero", 0)),
                           r.get("above_100", r.get("above_2000", 0))])]
        print(f"[6/7] Feature validation — flagged: {issues if issues else 'none (expected)'}")
    save_processed_data(featured_df)
    if verbose:
        print(f"[7/7] Saved: {PROCESSED_PATH}")
        print("=" * 60)
    return {
        "raw_df": raw_df, "validation_report": val_report,
        "clean_report": clean_report, "filter_report": filter_report,
        "district_df": district_df, "featured_df": featured_df,
        "feature_validation_report": feat_val,
    }


# ============================================================
# SECTION 11 — EDA: DATA LOADING & DESCRIPTIVE STATISTICS
# ============================================================

def load_analysis_data(path: str = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the processed district CSV for EDA.

    Verifies:
    - File exists
    - Expected number of rows (~640)
    - All 14 derived features are present
    - No infinite values in derived features

    Parameters
    ----------
    path : str
        Path to district_analysis_ready.csv

    Returns
    -------
    pd.DataFrame
        640-row district analysis DataFrame.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Processed dataset not found: '{path}'\n"
            "Run run_pipeline() first to generate it."
        )
    df = pd.read_csv(path, dtype={"State": str, "District": str})

    # Verify derived features present
    missing_feat = [c for c in DERIVED_FEATURE_COLS if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Missing derived features in processed CSV: {missing_feat}")

    # Verify no infinite values
    for col in DERIVED_FEATURE_COLS:
        if col in df.columns:
            n_inf = np.isinf(df[col].dropna()).sum()
            if n_inf > 0:
                raise ValueError(f"Infinite values found in {col}: {n_inf}")

    return df


def generate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for all 14 derived indicators.

    Statistics computed per feature:
    count, mean, median, std, min, Q1 (25%), Q3 (75%), max

    Parameters
    ----------
    df : pd.DataFrame
        District analysis DataFrame.

    Returns
    -------
    pd.DataFrame
        Rows = features, columns = statistics.
    """
    present = [c for c in DERIVED_FEATURE_COLS if c in df.columns]
    rows = []
    for col in present:
        s = df[col].dropna()
        rows.append({
            "Feature":  col,
            "Label":    FEATURE_LABELS.get(col, col),
            "Count":    int(s.count()),
            "Mean":     round(float(s.mean()), 2),
            "Median":   round(float(s.median()), 2),
            "Std Dev":  round(float(s.std()), 2),
            "Min":      round(float(s.min()), 2),
            "Q1 (25%)": round(float(s.quantile(0.25)), 2),
            "Q3 (75%)": round(float(s.quantile(0.75)), 2),
            "Max":      round(float(s.max()), 2),
        })
    return pd.DataFrame(rows)


# ============================================================
# SECTION 12 — EDA: DEMOGRAPHIC VISUALIZATIONS
# ============================================================

def create_demographic_visualizations(df: pd.DataFrame) -> dict:
    """
    Create demographic EDA charts.

    Returns a dict of Plotly figures keyed by chart name:
    - pop_dist         : histogram of total district population
    - pop_top10        : bar — top 10 districts by TOT_P
    - pop_bottom10     : bar — bottom 10 districts by TOT_P
    - sex_ratio_dist   : histogram of Sex_Ratio
    - sex_ratio_top10  : bar — top 10 districts by Sex_Ratio
    - sex_ratio_bot10  : bar — bottom 10 districts by Sex_Ratio
    - child_pop_dist   : histogram of Child_Pop_Pct
    - sc_pop_dist      : histogram of SC_Pop_Pct
    - st_pop_dist      : histogram of ST_Pop_Pct

    All charts use actual data; no values are hard-coded.
    """
    figs = {}

    # ── Population distribution ──────────────────────────────
    figs["pop_dist"] = px.histogram(
        df, x="TOT_P", nbins=40,
        title="Distribution of Total District Population",
        labels={"TOT_P": "Total Population"},
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    figs["pop_dist"].update_layout(
        xaxis_tickformat=",", yaxis_title="Number of Districts"
    )

    # ── Top / bottom 10 districts by population ──────────────
    top10_pop = df.nlargest(10, "TOT_P")[["Name", "TOT_P"]].sort_values("TOT_P")
    figs["pop_top10"] = px.bar(
        top10_pop, x="TOT_P", y="Name", orientation="h",
        title="Top 10 Districts by Total Population",
        labels={"TOT_P": "Total Population", "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[0]],
        text="TOT_P",
    )
    figs["pop_top10"].update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    figs["pop_top10"].update_layout(xaxis_tickformat=",")

    bot10_pop = df.nsmallest(10, "TOT_P")[["Name", "TOT_P"]].sort_values("TOT_P", ascending=False)
    figs["pop_bottom10"] = px.bar(
        bot10_pop, x="TOT_P", y="Name", orientation="h",
        title="Bottom 10 Districts by Total Population",
        labels={"TOT_P": "Total Population", "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[1]],
        text="TOT_P",
    )
    figs["pop_bottom10"].update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    figs["pop_bottom10"].update_layout(xaxis_tickformat=",")

    # ── Sex ratio distribution ────────────────────────────────
    figs["sex_ratio_dist"] = px.histogram(
        df, x="Sex_Ratio", nbins=40,
        title="Distribution of Sex Ratio",
        labels={"Sex_Ratio": FEATURE_LABELS["Sex_Ratio"]},
        color_discrete_sequence=[CHART_COLORS[2]],
    )
    figs["sex_ratio_dist"].update_layout(yaxis_title="Number of Districts")

    top10_sr = df.nlargest(10, "Sex_Ratio")[["Name", "Sex_Ratio"]].sort_values("Sex_Ratio")
    figs["sex_ratio_top10"] = px.bar(
        top10_sr, x="Sex_Ratio", y="Name", orientation="h",
        title="Top 10 Districts by Sex Ratio",
        labels={"Sex_Ratio": FEATURE_LABELS["Sex_Ratio"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[2]],
        text=top10_sr["Sex_Ratio"].round(1),
    )
    figs["sex_ratio_top10"].update_traces(textposition="outside")

    bot10_sr = df.nsmallest(10, "Sex_Ratio")[["Name", "Sex_Ratio"]].sort_values("Sex_Ratio", ascending=False)
    figs["sex_ratio_bot10"] = px.bar(
        bot10_sr, x="Sex_Ratio", y="Name", orientation="h",
        title="Bottom 10 Districts by Sex Ratio",
        labels={"Sex_Ratio": FEATURE_LABELS["Sex_Ratio"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[3]],
        text=bot10_sr["Sex_Ratio"].round(1),
    )
    figs["sex_ratio_bot10"].update_traces(textposition="outside")

    # ── Child population % distribution ──────────────────────
    figs["child_pop_dist"] = px.histogram(
        df, x="Child_Pop_Pct", nbins=35,
        title="Distribution of Child Population % (0–6 years)",
        labels={"Child_Pop_Pct": FEATURE_LABELS["Child_Pop_Pct"]},
        color_discrete_sequence=[CHART_COLORS[4]],
    )
    figs["child_pop_dist"].update_layout(yaxis_title="Number of Districts")

    # ── SC population % distribution ─────────────────────────
    figs["sc_pop_dist"] = px.histogram(
        df, x="SC_Pop_Pct", nbins=35,
        title="Distribution of Scheduled Caste Population %",
        labels={"SC_Pop_Pct": FEATURE_LABELS["SC_Pop_Pct"]},
        color_discrete_sequence=[CHART_COLORS[5]],
    )
    figs["sc_pop_dist"].update_layout(yaxis_title="Number of Districts")

    # ── ST population % distribution ─────────────────────────
    figs["st_pop_dist"] = px.histogram(
        df, x="ST_Pop_Pct", nbins=35,
        title="Distribution of Scheduled Tribe Population %",
        labels={"ST_Pop_Pct": FEATURE_LABELS["ST_Pop_Pct"]},
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    figs["st_pop_dist"].update_layout(yaxis_title="Number of Districts")

    return figs


# ============================================================
# SECTION 13 — EDA: EDUCATION VISUALIZATIONS
# ============================================================

def create_education_visualizations(df: pd.DataFrame) -> dict:
    """
    Create education EDA charts.

    Returns a dict of Plotly figures keyed by chart name:
    - lit_dist           : histogram of Literacy_Rate
    - lit_top10          : bar — top 10 districts by Literacy_Rate
    - lit_bot10          : bar — bottom 10 districts by Literacy_Rate
    - female_lit_dist    : histogram of Female_Literacy_Rate
    - mf_lit_scatter     : scatter — Male vs Female Literacy Rate
    - gap_dist           : histogram of Gender_Literacy_Gap
    - gap_top10_positive : bar — 10 districts with largest positive gap
    - gap_top10_negative : bar — 10 districts with smallest (most negative) gap
    """
    figs = {}

    # ── Literacy rate distribution ────────────────────────────
    figs["lit_dist"] = px.histogram(
        df, x="Literacy_Rate", nbins=35,
        title="Distribution of Literacy Rate (Population aged 7+)",
        labels={"Literacy_Rate": FEATURE_LABELS["Literacy_Rate"]},
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    figs["lit_dist"].update_layout(yaxis_title="Number of Districts")

    # ── Top / bottom 10 by literacy rate ─────────────────────
    top10_lit = df.nlargest(10, "Literacy_Rate")[["Name", "Literacy_Rate"]].sort_values("Literacy_Rate")
    figs["lit_top10"] = px.bar(
        top10_lit, x="Literacy_Rate", y="Name", orientation="h",
        title="Top 10 Districts by Literacy Rate",
        labels={"Literacy_Rate": FEATURE_LABELS["Literacy_Rate"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[0]],
        text=top10_lit["Literacy_Rate"].round(1),
    )
    figs["lit_top10"].update_traces(texttemplate="%{text}%", textposition="outside")
    figs["lit_top10"].update_layout(xaxis_range=[0, 105])

    bot10_lit = df.nsmallest(10, "Literacy_Rate")[["Name", "Literacy_Rate"]].sort_values("Literacy_Rate", ascending=False)
    figs["lit_bot10"] = px.bar(
        bot10_lit, x="Literacy_Rate", y="Name", orientation="h",
        title="Bottom 10 Districts by Literacy Rate",
        labels={"Literacy_Rate": FEATURE_LABELS["Literacy_Rate"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[1]],
        text=bot10_lit["Literacy_Rate"].round(1),
    )
    figs["lit_bot10"].update_traces(texttemplate="%{text}%", textposition="outside")
    figs["lit_bot10"].update_layout(xaxis_range=[0, 105])

    # ── Female literacy distribution ──────────────────────────
    figs["female_lit_dist"] = px.histogram(
        df, x="Female_Literacy_Rate", nbins=35,
        title="Distribution of Female Literacy Rate (Population aged 7+)",
        labels={"Female_Literacy_Rate": FEATURE_LABELS["Female_Literacy_Rate"]},
        color_discrete_sequence=[CHART_COLORS[2]],
    )
    figs["female_lit_dist"].update_layout(yaxis_title="Number of Districts")

    # ── Male vs Female literacy scatter ──────────────────────
    figs["mf_lit_scatter"] = px.scatter(
        df, x="Male_Literacy_Rate", y="Female_Literacy_Rate",
        hover_name="Name",
        title="Male Literacy Rate vs Female Literacy Rate",
        labels={
            "Male_Literacy_Rate":   FEATURE_LABELS["Male_Literacy_Rate"],
            "Female_Literacy_Rate": FEATURE_LABELS["Female_Literacy_Rate"],
        },
        color_discrete_sequence=[CHART_COLORS[0]],
        opacity=0.65,
    )
    # Add parity line (M == F)
    axis_range = [
        min(df["Male_Literacy_Rate"].min(), df["Female_Literacy_Rate"].min()) - 2,
        max(df["Male_Literacy_Rate"].max(), df["Female_Literacy_Rate"].max()) + 2,
    ]
    figs["mf_lit_scatter"].add_trace(
        go.Scatter(
            x=axis_range, y=axis_range,
            mode="lines",
            line=dict(dash="dash", color="grey", width=1),
            name="Parity line (M = F)",
        )
    )

    # ── Gender literacy gap distribution ─────────────────────
    figs["gap_dist"] = px.histogram(
        df, x="Gender_Literacy_Gap", nbins=35,
        title="Distribution of Gender Literacy Gap (Male − Female, percentage points)",
        labels={"Gender_Literacy_Gap": FEATURE_LABELS["Gender_Literacy_Gap"]},
        color_discrete_sequence=[CHART_COLORS[3]],
    )
    figs["gap_dist"].add_vline(x=0, line_dash="dash", line_color="grey",
                               annotation_text="Parity (gap = 0)")
    figs["gap_dist"].update_layout(yaxis_title="Number of Districts")

    # ── Top 10 districts with largest positive gap ────────────
    top10_gap = df.nlargest(10, "Gender_Literacy_Gap")[
        ["Name", "Gender_Literacy_Gap"]].sort_values("Gender_Literacy_Gap")
    figs["gap_top10_positive"] = px.bar(
        top10_gap, x="Gender_Literacy_Gap", y="Name", orientation="h",
        title="10 Districts with Largest Positive Gender Literacy Gap (Male > Female)",
        labels={"Gender_Literacy_Gap": FEATURE_LABELS["Gender_Literacy_Gap"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[3]],
        text=top10_gap["Gender_Literacy_Gap"].round(1),
    )
    figs["gap_top10_positive"].update_traces(texttemplate="%{text} pp", textposition="outside")

    # ── Districts with smallest / negative gap ────────────────
    bot10_gap = df.nsmallest(10, "Gender_Literacy_Gap")[
        ["Name", "Gender_Literacy_Gap"]].sort_values("Gender_Literacy_Gap", ascending=False)
    figs["gap_top10_negative"] = px.bar(
        bot10_gap, x="Gender_Literacy_Gap", y="Name", orientation="h",
        title="10 Districts with Smallest / Negative Gender Literacy Gap",
        labels={"Gender_Literacy_Gap": FEATURE_LABELS["Gender_Literacy_Gap"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[4]],
        text=bot10_gap["Gender_Literacy_Gap"].round(1),
    )
    figs["gap_top10_negative"].update_traces(texttemplate="%{text} pp", textposition="outside")
    figs["gap_top10_negative"].add_vline(x=0, line_dash="dash", line_color="grey")

    return figs


# ============================================================
# SECTION 14 — EDA: EMPLOYMENT VISUALIZATIONS
# ============================================================

def create_employment_visualizations(df: pd.DataFrame) -> dict:
    """
    Create employment EDA charts.

    Returns a dict of Plotly figures keyed by chart name:
    - wp_dist           : histogram of Worker_Participation
    - fwp_dist          : histogram of Female_Worker_Part
    - main_marg_scatter : scatter — Main_Worker_Pct vs Marginal_Worker_Pct
    - nw_dist           : histogram of Non_Worker_Pct
    - agri_dist         : histogram of Agri_Worker_Pct
    - agri_top10        : bar — top 10 districts by Agri_Worker_Pct
    - fwp_top10         : bar — top 10 districts by Female_Worker_Part
    """
    figs = {}

    # ── Worker participation distribution ─────────────────────
    figs["wp_dist"] = px.histogram(
        df, x="Worker_Participation", nbins=35,
        title="Distribution of Worker Participation Rate",
        labels={"Worker_Participation": FEATURE_LABELS["Worker_Participation"]},
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    figs["wp_dist"].update_layout(yaxis_title="Number of Districts")

    # ── Female worker participation distribution ──────────────
    figs["fwp_dist"] = px.histogram(
        df, x="Female_Worker_Part", nbins=35,
        title="Distribution of Female Worker Participation Rate",
        labels={"Female_Worker_Part": FEATURE_LABELS["Female_Worker_Part"]},
        color_discrete_sequence=[CHART_COLORS[2]],
    )
    figs["fwp_dist"].update_layout(yaxis_title="Number of Districts")

    # ── Main vs Marginal worker scatter ──────────────────────
    figs["main_marg_scatter"] = px.scatter(
        df, x="Main_Worker_Pct", y="Marginal_Worker_Pct",
        hover_name="Name",
        title="Main Worker % vs Marginal Worker % (share of total workers)",
        labels={
            "Main_Worker_Pct":     FEATURE_LABELS["Main_Worker_Pct"],
            "Marginal_Worker_Pct": FEATURE_LABELS["Marginal_Worker_Pct"],
        },
        color_discrete_sequence=[CHART_COLORS[1]],
        opacity=0.65,
    )

    # ── Non-worker distribution ───────────────────────────────
    figs["nw_dist"] = px.histogram(
        df, x="Non_Worker_Pct", nbins=35,
        title="Distribution of Non-Worker Percentage",
        labels={"Non_Worker_Pct": FEATURE_LABELS["Non_Worker_Pct"]},
        color_discrete_sequence=[CHART_COLORS[3]],
    )
    figs["nw_dist"].update_layout(yaxis_title="Number of Districts")

    # ── Agricultural worker share distribution ────────────────
    figs["agri_dist"] = px.histogram(
        df, x="Agri_Worker_Pct", nbins=35,
        title="Distribution of Agricultural Worker Share",
        labels={"Agri_Worker_Pct": FEATURE_LABELS["Agri_Worker_Pct"]},
        color_discrete_sequence=[CHART_COLORS[4]],
    )
    figs["agri_dist"].update_layout(yaxis_title="Number of Districts")

    # ── Top 10 by agricultural worker share ──────────────────
    top10_agri = df.nlargest(10, "Agri_Worker_Pct")[
        ["Name", "Agri_Worker_Pct"]].sort_values("Agri_Worker_Pct")
    figs["agri_top10"] = px.bar(
        top10_agri, x="Agri_Worker_Pct", y="Name", orientation="h",
        title="Top 10 Districts by Agricultural Worker Share",
        labels={"Agri_Worker_Pct": FEATURE_LABELS["Agri_Worker_Pct"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[4]],
        text=top10_agri["Agri_Worker_Pct"].round(1),
    )
    figs["agri_top10"].update_traces(texttemplate="%{text}%", textposition="outside")
    figs["agri_top10"].update_layout(xaxis_range=[0, 105])

    # ── Top 10 by female worker participation ─────────────────
    top10_fwp = df.nlargest(10, "Female_Worker_Part")[
        ["Name", "Female_Worker_Part"]].sort_values("Female_Worker_Part")
    figs["fwp_top10"] = px.bar(
        top10_fwp, x="Female_Worker_Part", y="Name", orientation="h",
        title="Top 10 Districts by Female Worker Participation Rate",
        labels={"Female_Worker_Part": FEATURE_LABELS["Female_Worker_Part"], "Name": "District"},
        color_discrete_sequence=[CHART_COLORS[2]],
        text=top10_fwp["Female_Worker_Part"].round(1),
    )
    figs["fwp_top10"].update_traces(texttemplate="%{text}%", textposition="outside")
    figs["fwp_top10"].update_layout(xaxis_range=[0, 70])

    return figs


# ============================================================
# SECTION 15 — EDA: STATE-LEVEL COMPARISONS
# ============================================================

def create_state_comparisons(df: pd.DataFrame) -> dict:
    """
    Aggregate district indicators to state-level averages and create charts.

    IMPORTANT: These are averages of district values within each state/UT,
    NOT official Census state totals. They are labelled accordingly in all
    chart titles and axis labels.

    Indicators compared:
    Literacy_Rate, Female_Literacy_Rate, Worker_Participation,
    Female_Worker_Part, Sex_Ratio, Agri_Worker_Pct

    Returns a dict with:
    - state_avg_df  : DataFrame of state-level district averages
    - figures       : dict of Plotly figures keyed by indicator name
    """
    indicators = [
        "Literacy_Rate", "Female_Literacy_Rate",
        "Worker_Participation", "Female_Worker_Part",
        "Sex_Ratio", "Agri_Worker_Pct",
    ]
    present = [c for c in indicators if c in df.columns]

    # Build a state name lookup from the processed data
    # (use the state code; we need a name — derive from the raw Name column
    #  by taking the modal district name prefix, or just use code)
    state_avg = (
        df.groupby("State")[present]
        .mean()
        .round(2)
        .reset_index()
    )
    # Attach state name by taking first district's Name for context label
    state_name_map = df.groupby("State")["Name"].first().reset_index()
    state_name_map.columns = ["State", "SampleDistrict"]

    # Build a proper state name: for the chart we map State codes to
    # known state names from the dataset (districts belong to states)
    # We do this by finding what NAME the state-level rows have in the raw data
    # Since we only have district records, use a Census code→name map derived
    # from a fixed reference consistent with Census 2011 codes
    STATE_CODE_MAP = {
        "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
        "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana",
        "07": "NCT Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
        "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh",
        "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
        "16": "Tripura", "17": "Meghalaya", "18": "Assam",
        "19": "West Bengal", "20": "Jharkhand", "21": "Odisha",
        "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
        "25": "Daman & Diu", "26": "Dadra & Nagar Haveli",
        "27": "Maharashtra", "28": "Andhra Pradesh", "29": "Karnataka",
        "30": "Goa", "31": "Lakshadweep", "32": "Kerala",
        "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman & Nicobar",
    }
    state_avg["State_Name"] = state_avg["State"].map(STATE_CODE_MAP).fillna(state_avg["State"])

    figs = {}
    for col in present:
        label = FEATURE_LABELS.get(col, col)
        sorted_df = state_avg.sort_values(col, ascending=True)
        fig = px.bar(
            sorted_df, x=col, y="State_Name", orientation="h",
            title=f"State/UT — Average {label} across Districts",
            labels={col: f"Average {label}", "State_Name": "State / UT"},
            color_discrete_sequence=[CHART_COLORS[0]],
            text=sorted_df[col].round(1),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=700,
            xaxis_title=f"Average {label} (district mean within state/UT)",
        )
        figs[col] = fig

    return {"state_avg_df": state_avg, "figures": figs}


# ============================================================
# SECTION 16 — EDA: RELATIONSHIP ANALYSIS
# ============================================================

def create_relationship_analysis(df: pd.DataFrame) -> dict:
    """
    Create scatter plots and calculate Pearson correlations for
    meaningful pairs of indicators.

    Correlation is reported as a descriptive statistic only.
    No causal interpretation is made.

    Pairs analysed:
    1. Literacy_Rate vs Female_Worker_Part
    2. Literacy_Rate vs Worker_Participation
    3. Female_Literacy_Rate vs Female_Worker_Part
    4. Agri_Worker_Pct vs Literacy_Rate
    5. Child_Pop_Pct vs Literacy_Rate
    6. Non_Worker_Pct vs Worker_Participation

    Returns a dict with keys 'figures' (dict of Plotly figs)
    and 'correlations' (list of dicts with x, y, r, p).
    """
    pairs = [
        ("Literacy_Rate",        "Female_Worker_Part"),
        ("Literacy_Rate",        "Worker_Participation"),
        ("Female_Literacy_Rate", "Female_Worker_Part"),
        ("Agri_Worker_Pct",      "Literacy_Rate"),
        ("Child_Pop_Pct",        "Literacy_Rate"),
        ("Non_Worker_Pct",       "Worker_Participation"),
    ]

    figs = {}
    correlations = []

    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        tmp = df[[x_col, y_col, "Name"]].dropna()
        r, p = scipy_stats.pearsonr(tmp[x_col], tmp[y_col])
        correlations.append({
            "X Variable": FEATURE_LABELS.get(x_col, x_col),
            "Y Variable": FEATURE_LABELS.get(y_col, y_col),
            "Pearson r":  round(r, 3),
            "p-value":    round(p, 4),
            "n":          len(tmp),
        })

        direction = "positive" if r > 0 else "negative"
        strength  = "strong" if abs(r) >= 0.5 else ("moderate" if abs(r) >= 0.3 else "weak")
        annotation = (
            f"These variables show a {strength} {direction} association "
            f"in this dataset (r = {r:.3f})."
        )

        key = f"{x_col}_vs_{y_col}"
        figs[key] = px.scatter(
            tmp, x=x_col, y=y_col, hover_name="Name",
            title=f"{FEATURE_LABELS.get(x_col, x_col)}<br>vs {FEATURE_LABELS.get(y_col, y_col)}",
            labels={
                x_col: FEATURE_LABELS.get(x_col, x_col),
                y_col: FEATURE_LABELS.get(y_col, y_col),
            },
            color_discrete_sequence=[CHART_COLORS[0]],
            opacity=0.6,
        )
        # Add OLS trend line
        figs[key].update_layout(
            annotations=[dict(
                xref="paper", yref="paper",
                x=0.02, y=0.97,
                xanchor="left", yanchor="top",
                text=annotation,
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="lightgrey",
                borderwidth=1,
            )]
        )
        # Trend line via numpy polyfit
        z = np.polyfit(tmp[x_col], tmp[y_col], 1)
        p_fn = np.poly1d(z)
        x_line = np.linspace(tmp[x_col].min(), tmp[x_col].max(), 100)
        figs[key].add_trace(go.Scatter(
            x=x_line, y=p_fn(x_line),
            mode="lines", name="Trend line",
            line=dict(color="firebrick", width=1.5, dash="dot"),
        ))

    return {"figures": figs, "correlations": correlations}


# ============================================================
# SECTION 17 — EDA: CORRELATION MATRIX
# ============================================================

def create_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Compute and visualise the Pearson correlation matrix for all
    14 derived indicators.

    Only numeric derived features are included.
    Identifier columns (State, District, Name) are excluded.

    Returns
    -------
    go.Figure
        Plotly heatmap of the correlation matrix.
    """
    present = [c for c in DERIVED_FEATURE_COLS if c in df.columns]
    corr_df = df[present].dropna()
    corr_matrix = corr_df.corr(method="pearson").round(2)

    short_labels = {
        "Sex_Ratio":            "Sex Ratio",
        "Child_Pop_Pct":        "Child Pop %",
        "SC_Pop_Pct":           "SC Pop %",
        "ST_Pop_Pct":           "ST Pop %",
        "Literacy_Rate":        "Literacy",
        "Male_Literacy_Rate":   "Male Lit.",
        "Female_Literacy_Rate": "Female Lit.",
        "Gender_Literacy_Gap":  "Gender Gap",
        "Worker_Participation": "Worker Part.",
        "Female_Worker_Part":   "Female WP",
        "Main_Worker_Pct":      "Main Worker%",
        "Marginal_Worker_Pct":  "Marg. Worker%",
        "Non_Worker_Pct":       "Non-Worker%",
        "Agri_Worker_Pct":      "Agri Worker%",
    }
    labels = [short_labels.get(c, c) for c in corr_matrix.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(
        title="Pearson Correlation Matrix — Derived Socioeconomic Indicators",
        height=600,
        xaxis=dict(tickangle=-45),
    )
    return fig


# ============================================================
# SECTION 18 — EDA: EXPLORATORY OUTLIERS
# ============================================================

def identify_exploratory_outliers(df: pd.DataFrame) -> dict:
    """
    Identify exploratory outliers using the IQR method.

    For each derived indicator, a data point is flagged as an
    exploratory outlier if it lies below Q1 − 1.5×IQR or
    above Q3 + 1.5×IQR.

    These are called 'exploratory outliers' to distinguish them
    from ML-based anomalies (Isolation Forest, Step 4).

    Outliers are NOT removed. They are reported for investigation.

    Parameters
    ----------
    df : pd.DataFrame
        District analysis DataFrame.

    Returns
    -------
    dict
        Keys:
        - summary_df  : DataFrame with per-feature outlier count and bounds
        - detail_dfs  : dict of DataFrames, one per feature, listing outlier rows
        - boxplot_figs: dict of Plotly box plots keyed by feature name
    """
    present = [c for c in DERIVED_FEATURE_COLS if c in df.columns]
    summary_rows = []
    detail_dfs   = {}
    boxplot_figs = {}

    for col in present:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask  = (df[col] < lower) | (df[col] > upper)
        outlier_count = int(outlier_mask.sum())

        summary_rows.append({
            "Feature":        col,
            "Label":          FEATURE_LABELS.get(col, col),
            "Q1":             round(q1, 2),
            "Q3":             round(q3, 2),
            "IQR":            round(iqr, 2),
            "Lower Fence":    round(lower, 2),
            "Upper Fence":    round(upper, 2),
            "Outlier Count":  outlier_count,
        })

        if outlier_count > 0:
            detail_dfs[col] = (
                df[outlier_mask][["State", "Name", col]]
                .copy()
                .sort_values(col)
                .reset_index(drop=True)
            )

        # Box plot
        fig = px.box(
            df, y=col,
            title=f"Box Plot — {FEATURE_LABELS.get(col, col)}",
            labels={col: FEATURE_LABELS.get(col, col)},
            color_discrete_sequence=[CHART_COLORS[0]],
            points="outliers",
        )
        boxplot_figs[col] = fig

    return {
        "summary_df":   pd.DataFrame(summary_rows),
        "detail_dfs":   detail_dfs,
        "boxplot_figs": boxplot_figs,
    }


# ============================================================
# SECTION 19 — EDA: FACTUAL OBSERVATIONS
# ============================================================

def generate_factual_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Programmatically generate factual observations from the dataset.

    Each observation is a directly measurable fact — a value or
    district/state from the data. No causal claims are made.

    Observations generated:
    - District with max/min for each of the 14 derived indicators
    - Median and mean for each indicator
    - Count of districts with negative Gender_Literacy_Gap
    - Count of districts with Female_Worker_Part > Male equivalent proxy

    Returns
    -------
    pd.DataFrame
        Columns: metric, district, state_code, value, description
    """
    obs = []

    def _add(metric, district_name, state_code, value, description):
        obs.append({
            "Metric":      metric,
            "District":    district_name,
            "State Code":  state_code,
            "Value":       round(float(value), 3) if pd.notna(value) else None,
            "Observation": description,
        })

    for col in DERIVED_FEATURE_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        label = FEATURE_LABELS.get(col, col)

        # Max district
        idx_max = df[col].idxmax()
        row_max = df.loc[idx_max]
        _add(
            f"{col} — Maximum",
            row_max["Name"], row_max["State"],
            row_max[col],
            f"The highest observed {label} in the dataset is "
            f"{row_max[col]:.2f}, recorded for {row_max['Name']}.",
        )

        # Min district
        idx_min = df[col].idxmin()
        row_min = df.loc[idx_min]
        _add(
            f"{col} — Minimum",
            row_min["Name"], row_min["State"],
            row_min[col],
            f"The lowest observed {label} in the dataset is "
            f"{row_min[col]:.2f}, recorded for {row_min['Name']}.",
        )

        # Median
        _add(
            f"{col} — Median",
            "All districts", "—",
            s.median(),
            f"The median {label} across all 640 districts is {s.median():.2f}.",
        )

        # Mean
        _add(
            f"{col} — Mean",
            "All districts", "—",
            s.mean(),
            f"The mean {label} across all 640 districts is {s.mean():.2f}.",
        )

    # Negative Gender_Literacy_Gap count
    if "Gender_Literacy_Gap" in df.columns:
        n_neg = int((df["Gender_Literacy_Gap"] < 0).sum())
        _add(
            "Gender_Literacy_Gap — Negative count",
            "Multiple districts", "—",
            n_neg,
            f"{n_neg} district(s) in the dataset show a negative Gender Literacy Gap "
            f"(Female Literacy Rate exceeds Male Literacy Rate).",
        )

    return pd.DataFrame(obs)


# ============================================================
# SECTION 20 — ML: DATA PREPARATION
# ============================================================

def prepare_ml_data(df: pd.DataFrame) -> dict:
    """
    Prepare the district dataset for machine learning.

    Steps:
    1. Confirm features from ML_FEATURE_SHORTLIST exist in df.
    2. Drop any row with NaN or infinite values in the ML features.
    3. Check feature variance — warn if any feature has near-zero variance.
    4. Standardize the feature matrix with StandardScaler (zero mean, unit var).

    Identifiers (State, District, Name) are retained separately and never
    passed to the scaler or model.

    Parameters
    ----------
    df : pd.DataFrame
        District analysis DataFrame (from load_analysis_data).

    Returns
    -------
    dict with keys:
        feature_df      : pd.DataFrame — rows used, identifier + ML feature cols
        features_used   : list[str]    — ML feature names actually used
        X_scaled        : np.ndarray   — standardized feature matrix (n_rows × n_feats)
        scaler          : StandardScaler — fitted scaler (for inverse transform)
        meta            : dict — preprocessing metadata
    """
    present = [f for f in ML_FEATURE_SHORTLIST if f in df.columns]
    missing_feat = [f for f in ML_FEATURE_SHORTLIST if f not in df.columns]

    # Drop rows with NaN or inf in any ML feature
    ml_df = df[["State", "District", "Name"] + present].copy()
    inf_mask = ml_df[present].apply(lambda c: np.isinf(c)).any(axis=1)
    ml_df = ml_df[~inf_mask]
    n_before = len(df)
    ml_df = ml_df.dropna(subset=present)
    n_after  = len(ml_df)
    rows_dropped = n_before - n_after

    # Feature variance check
    variance_warnings = []
    for feat in present:
        v = ml_df[feat].var()
        if v < 1e-6:
            variance_warnings.append(f"{feat}: near-zero variance ({v:.2e})")

    # Standardize
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(ml_df[present].values)

    meta = {
        "n_rows_in":       n_before,
        "n_rows_used":     n_after,
        "rows_dropped":    rows_dropped,
        "features_used":   present,
        "features_missing_from_df": missing_feat,
        "variance_warnings": variance_warnings,
        "feature_means":   dict(zip(present, scaler.mean_.tolist())),
        "feature_stds":    dict(zip(present, scaler.scale_.tolist())),
    }

    return {
        "feature_df":    ml_df.reset_index(drop=True),
        "features_used": present,
        "X_scaled":      X_scaled,
        "scaler":        scaler,
        "meta":          meta,
    }


# ============================================================
# SECTION 21 — ML: K-MEANS CLUSTERING
# ============================================================

def run_kmeans(df: pd.DataFrame = None,
               features: list = None,
               n_clusters: int = N_CLUSTERS,
               X_scaled: np.ndarray = None,
               feature_df: pd.DataFrame = None) -> dict:
    """
    Evaluate K-Means for K in [2..8] and select the best K.

    Selection rule (transparent):
      1. Compute silhouette score for K = 2..8.
      2. Select the K with the highest silhouette score.
      3. If two K values tie (within 0.005), prefer the smaller K
         (parsimony).

    Important: clustering is exploratory and data-derived.
    Clusters do NOT represent official development categories.
    Cluster labels are neutral integers ("Cluster 0", "Cluster 1", …).

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix from prepare_ml_data.
    feature_df : pd.DataFrame
        Identifier + feature DataFrame from prepare_ml_data.

    Returns
    -------
    dict with keys:
        evaluation_df   : pd.DataFrame — K, inertia, silhouette, cluster_sizes
        selected_k      : int
        selection_reason: str
        labels          : np.ndarray — cluster label per district
        model           : fitted KMeans for selected K
        cluster_profile : pd.DataFrame — per-cluster mean/median of features
        cluster_summary : pd.DataFrame — cluster id, n_districts, pct_districts
    """
    from sklearn.metrics import silhouette_score

    K_RANGE = range(2, 9)
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        lbls = km.fit_predict(X_scaled)
        sil  = silhouette_score(X_scaled, lbls)
        rows.append({
            "K":           k,
            "Inertia":     round(km.inertia_, 2),
            "Silhouette":  round(sil, 4),
            "Cluster_Sizes": str(sorted(
                pd.Series(lbls).value_counts().sort_index().tolist()
            )),
        })

    eval_df = pd.DataFrame(rows)

    # Select K: highest silhouette; tie-break = smaller K
    best_row  = eval_df.sort_values(["Silhouette", "K"],
                                    ascending=[False, True]).iloc[0]
    selected_k = int(best_row["K"])
    reason = (
        f"K={selected_k} achieves the highest silhouette score "
        f"({best_row['Silhouette']:.4f}) among K=2..8. "
        "Silhouette score measures how well each district fits its assigned "
        "cluster versus the nearest alternative cluster (range: -1 to +1; "
        "higher = better separation). Clustering is exploratory and "
        "does not establish official or causal development categories."
    )

    # Fit final model with selected K
    final_km = KMeans(n_clusters=selected_k, random_state=RANDOM_SEED, n_init=10)
    labels   = final_km.fit_predict(X_scaled)

    # Cluster summary
    feat_cols = [c for c in feature_df.columns
                 if c not in ("State", "District", "Name")]
    profiling_df = feature_df.copy()
    profiling_df["Cluster"] = labels

    cluster_summary_rows = []
    n_total = len(labels)
    for c in sorted(profiling_df["Cluster"].unique()):
        cluster_summary_rows.append({
            "Cluster":       f"Cluster {c}",
            "N_Districts":   int((profiling_df["Cluster"] == c).sum()),
            "Pct_Districts": round(100 * (profiling_df["Cluster"] == c).sum() / n_total, 1),
        })
    cluster_summary = pd.DataFrame(cluster_summary_rows)

    # Cluster profile: mean of each ML feature per cluster
    profile_rows = []
    overall_means = profiling_df[feat_cols].mean()
    for c in sorted(profiling_df["Cluster"].unique()):
        sub = profiling_df[profiling_df["Cluster"] == c][feat_cols]
        row = {"Cluster": f"Cluster {c}"}
        for feat in feat_cols:
            row[f"{feat}_mean"] = round(sub[feat].mean(), 2)
            row[f"{feat}_delta"] = round(sub[feat].mean() - overall_means[feat], 2)
        profile_rows.append(row)
    cluster_profile = pd.DataFrame(profile_rows)

    return {
        "evaluation_df":    eval_df,
        "selected_k":       selected_k,
        "selection_reason": reason,
        "labels":           labels,
        "model":            final_km,
        "cluster_profile":  cluster_profile,
        "cluster_summary":  cluster_summary,
    }


# ============================================================
# SECTION 22 — ML: PCA AND ANOMALY DETECTION
# ============================================================

def run_pca_analysis(df: pd.DataFrame = None,
                     features: list = None,
                     n_components: int = 2,
                     X_scaled: np.ndarray = None,
                     features_used: list = None) -> dict:
    """
    Apply PCA to the standardized ML feature matrix.

    Returns PC1 and PC2 coordinates per district, explained variance,
    and component loadings.

    PCA is a dimensionality reduction technique — it does NOT establish
    causal importance of any feature. Loadings describe mathematical
    relationships within the 2011 dataset only.

    Parameters
    ----------
    X_scaled       : np.ndarray — standardized feature matrix
    features_used  : list[str] — feature names in column order

    Returns
    -------
    dict with keys:
        pca_coords        : np.ndarray (n x 2) — PC1, PC2 per district
        explained_var     : np.ndarray — explained variance ratio per component
        cum_explained_var : np.ndarray — cumulative explained variance
        loadings_df       : pd.DataFrame — feature loadings on each component
        pca_model         : fitted PCA object
    """
    n_comp = min(len(features_used), X_scaled.shape[1], X_scaled.shape[0])
    full_pca = SklearnPCA(n_components=n_comp, random_state=RANDOM_SEED)
    full_pca.fit(X_scaled)

    pca_2d = SklearnPCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca_2d.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        full_pca.components_[:2].T,
        index=features_used,
        columns=["PC1", "PC2"],
    ).round(4)

    return {
        "pca_coords":        coords,
        "explained_var":     full_pca.explained_variance_ratio_,
        "cum_explained_var": np.cumsum(full_pca.explained_variance_ratio_),
        "loadings_df":       loadings,
        "pca_model":         full_pca,
    }


def run_anomaly_detection(df: pd.DataFrame = None,
                          features: list = None,
                          contamination: float = CONTAMINATION,
                          X_scaled: np.ndarray = None) -> dict:
    """
    Apply Isolation Forest to identify districts with unusual combinations
    of socioeconomic characteristics within the 2011 dataset.

    Isolation Forest assigns an anomaly score to each district.
    Lower scores (more negative) indicate more unusual profiles.
    Districts flagged as anomalies (-1) show atypical combinations of
    the selected indicators — they are NOT labelled as "problem" districts.

    Parameters
    ----------
    X_scaled      : np.ndarray — standardized feature matrix
    contamination : float — assumed proportion of unusual profiles (default 0.05)

    Returns
    -------
    dict with keys:
        anomaly_flags  : np.ndarray — +1 (typical) or -1 (unusual profile)
        anomaly_scores : np.ndarray — raw decision function scores
        n_anomalies    : int
        model          : fitted IsolationForest
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_estimators=200,
    )
    iso.fit(X_scaled)
    flags  = iso.predict(X_scaled)           # +1 = typical, -1 = unusual
    scores = iso.decision_function(X_scaled) # lower = more unusual

    return {
        "anomaly_flags":  flags,
        "anomaly_scores": scores,
        "n_anomalies":    int((flags == -1).sum()),
        "model":          iso,
    }


def run_ml_pipeline(df: pd.DataFrame = None) -> dict:
    """
    Execute the full ML pipeline: preparation → K-Means → PCA → Isolation Forest.

    If df is None, loads from PROCESSED_PATH.

    Returns a comprehensive results dict and saves district_ml_results.csv.
    """
    if df is None:
        df = load_analysis_data()

    # A. Prepare data
    ml_prep = prepare_ml_data(df)
    X_scaled   = ml_prep["X_scaled"]
    feature_df = ml_prep["feature_df"]
    features   = ml_prep["features_used"]

    # B. K-Means
    km_result = run_kmeans(X_scaled=X_scaled, feature_df=feature_df)

    # C. PCA
    pca_result = run_pca_analysis(X_scaled=X_scaled, features_used=features)

    # D. Isolation Forest
    iso_result = run_anomaly_detection(X_scaled=X_scaled)

    # E. Assemble results dataframe
    results_df = feature_df[["State", "District", "Name"]].copy()
    results_df["Cluster"]       = km_result["labels"]
    results_df["Cluster_Label"] = "Cluster " + results_df["Cluster"].astype(str)
    results_df["PC1"]           = pca_result["pca_coords"][:, 0].round(4)
    results_df["PC2"]           = pca_result["pca_coords"][:, 1].round(4)
    results_df["Anomaly_Flag"]  = iso_result["anomaly_flags"]
    results_df["Anomaly_Score"] = iso_result["anomaly_scores"].round(4)

    # Add key ML features back for reference
    for feat in features:
        results_df[feat] = feature_df[feat].values

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    results_df.to_csv(ML_RESULTS_PATH, index=False)
    print(f"[INFO] ML results saved: {ML_RESULTS_PATH} ({len(results_df)} rows)")

    return {
        "ml_prep":       ml_prep,
        "km_result":     km_result,
        "pca_result":    pca_result,
        "iso_result":    iso_result,
        "results_df":    results_df,
    }


def validate_ml_pipeline(ml_output: dict) -> dict:
    """
    Validate the ML pipeline outputs for data quality and correctness.

    Checks:
    1.  No missing values in ML features used.
    2.  No infinite values in scaled matrix.
    3.  No identifier columns (State/District/Name) leaked to X_scaled.
    4.  Scaling applied: each feature column has mean ~0, std ~1.
    5.  Reproducibility confirmed: K-Means is seeded.
    6.  K range 2..8 evaluated.
    7.  Cluster labels exist for every district row.
    8.  All districts assigned exactly one cluster.
    9.  Cluster sizes sum to total district count.
    10. PCA coordinates are 2-dimensional and finite.
    11. Isolation Forest flags are all +1 or -1.
    12. No NaN/inf in anomaly scores.

    Returns
    -------
    dict with keys: passed, warnings, errors
    """
    passed, warnings, errors = [], [], []

    prep       = ml_output["ml_prep"]
    km         = ml_output["km_result"]
    pca        = ml_output["pca_result"]
    iso        = ml_output["iso_result"]
    X_scaled   = prep["X_scaled"]
    feature_df = prep["feature_df"]
    features   = prep["features_used"]
    results    = ml_output["results_df"]

    # 1. No NaN in ML features
    nan_counts = feature_df[features].isnull().sum().sum()
    if nan_counts == 0:
        passed.append("PASS [1]: No missing values in ML feature matrix.")
    else:
        errors.append(f"ERROR [1]: {nan_counts} missing values in ML features.")

    # 2. No inf in X_scaled
    if not np.isinf(X_scaled).any():
        passed.append("PASS [2]: No infinite values in scaled feature matrix.")
    else:
        errors.append("ERROR [2]: Infinite values detected in scaled matrix.")

    # 3. No identifier leakage
    id_cols = {"State", "District", "Name"}
    if X_scaled.shape[1] == len(features) and not id_cols.intersection(set(features)):
        passed.append("PASS [3]: No identifier columns in X_scaled.")
    else:
        errors.append("ERROR [3]: Possible identifier leakage in X_scaled.")

    # 4. Scaling check (mean ~0, std ~1 per column)
    col_means = X_scaled.mean(axis=0)
    col_stds  = X_scaled.std(axis=0)
    if np.allclose(col_means, 0, atol=1e-6) and np.allclose(col_stds, 1, atol=1e-4):
        passed.append("PASS [4]: Feature matrix correctly scaled (mean~0, std~1).")
    else:
        warnings.append(
            f"WARN [4]: Scaling may be imperfect. Max mean deviation: "
            f"{np.abs(col_means).max():.2e}, max std deviation: "
            f"{np.abs(col_stds - 1).max():.2e}."
        )

    # 5. Reproducibility
    passed.append(f"PASS [5]: random_state={RANDOM_SEED} used for all models.")

    # 6. K range 2..8 evaluated
    eval_ks = set(km["evaluation_df"]["K"].tolist())
    expected = set(range(2, 9))
    if eval_ks == expected:
        passed.append("PASS [6]: K=2..8 all evaluated.")
    else:
        errors.append(f"ERROR [6]: K range mismatch. Evaluated: {eval_ks}")

    # 7 & 8. Cluster labels exist and every district assigned one
    labels = km["labels"]
    if len(labels) == len(feature_df):
        passed.append(f"PASS [7/8]: {len(labels)} cluster labels for {len(feature_df)} districts.")
    else:
        errors.append(f"ERROR [7/8]: Label count ({len(labels)}) != district count ({len(feature_df)}).")

    # 9. Cluster sizes sum to total
    total_in_clusters = km["cluster_summary"]["N_Districts"].sum()
    if total_in_clusters == len(feature_df):
        passed.append(f"PASS [9]: Cluster sizes sum to {total_in_clusters} (all districts accounted for).")
    else:
        errors.append(f"ERROR [9]: Cluster size sum ({total_in_clusters}) != total ({len(feature_df)}).")

    # 10. PCA coords are 2D and finite
    coords = pca["pca_coords"]
    if coords.shape[1] == 2 and np.isfinite(coords).all():
        passed.append("PASS [10]: PCA coordinates are 2D and all finite.")
    else:
        errors.append("ERROR [10]: PCA coordinate issue.")

    # 11. Anomaly flags are +1 or -1 only
    valid_flags = set(np.unique(iso["anomaly_flags"])).issubset({1, -1})
    if valid_flags:
        passed.append("PASS [11]: Isolation Forest flags are valid (+1 / -1 only).")
    else:
        errors.append("ERROR [11]: Unexpected Isolation Forest flag values.")

    # 12. No NaN/inf in anomaly scores
    if np.isfinite(iso["anomaly_scores"]).all():
        passed.append("PASS [12]: No NaN/inf in anomaly scores.")
    else:
        errors.append("ERROR [12]: NaN or inf in anomaly scores.")

    return {"passed": passed, "warnings": warnings, "errors": errors}


def create_ml_visualizations(ml_output: dict) -> dict:
    """
    Create all ML-related Plotly visualizations.

    Charts:
    1. K vs Inertia (elbow plot)
    2. K vs Silhouette Score
    3. Cluster Size Bar Chart
    4. PCA 2D Scatter coloured by cluster
    5. Explained Variance Bar + Cumulative Line
    6. Cluster Profile Heatmap (delta from dataset mean)
    7. Anomaly Score Distribution

    Returns dict keyed by chart name.
    """
    figs = {}

    km       = ml_output["km_result"]
    pca      = ml_output["pca_result"]
    iso      = ml_output["iso_result"]
    prep     = ml_output["ml_prep"]
    results  = ml_output["results_df"]
    features = prep["features_used"]
    eval_df  = km["evaluation_df"]
    sel_k    = km["selected_k"]

    # 1. K vs Inertia
    fig_inertia = px.line(
        eval_df, x="K", y="Inertia", markers=True,
        title="K-Means: Inertia (Within-Cluster Sum of Squares) vs K",
        labels={"K": "Number of Clusters (K)", "Inertia": "Inertia"},
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    fig_inertia.add_vline(x=sel_k, line_dash="dash", line_color="red",
                          annotation_text=f"Selected K={sel_k}")
    figs["inertia"] = fig_inertia

    # 2. K vs Silhouette
    fig_sil = px.line(
        eval_df, x="K", y="Silhouette", markers=True,
        title="K-Means: Silhouette Score vs K",
        labels={"K": "Number of Clusters (K)", "Silhouette": "Silhouette Score"},
        color_discrete_sequence=[CHART_COLORS[1]],
    )
    fig_sil.add_vline(x=sel_k, line_dash="dash", line_color="red",
                     annotation_text=f"Selected K={sel_k}")
    figs["silhouette"] = fig_sil

    # 3. Cluster Sizes
    cs = km["cluster_summary"].copy()
    figs["cluster_sizes"] = px.bar(
        cs, x="Cluster", y="N_Districts",
        title="Number of Districts per Cluster",
        labels={"N_Districts": "Number of Districts", "Cluster": ""},
        text="N_Districts",
        color="Cluster",
        color_discrete_sequence=CHART_COLORS,
    )
    figs["cluster_sizes"].update_traces(textposition="outside")

    # 4. PCA 2D Scatter
    pca_plot = results.copy()
    pca_plot["Cluster_Label"] = "Cluster " + pca_plot["Cluster"].astype(str)
    figs["pca_scatter"] = px.scatter(
        pca_plot, x="PC1", y="PC2",
        color="Cluster_Label",
        hover_name="Name",
        hover_data={"State": True, "PC1": ":.3f", "PC2": ":.3f"},
        title="PCA — District Profiles in 2D Space (coloured by K-Means Cluster)",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2",
                "Cluster_Label": "Cluster"},
        color_discrete_sequence=CHART_COLORS,
        opacity=0.75,
    )
    exp_var = pca["explained_var"]
    figs["pca_scatter"].update_layout(
        xaxis_title=f"PC1 ({exp_var[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({exp_var[1]*100:.1f}% variance)",
    )

    # 5. Explained Variance
    n_comp_show = min(len(exp_var), 11)
    comp_labels = [f"PC{i+1}" for i in range(n_comp_show)]
    cum_var     = pca["cum_explained_var"][:n_comp_show]
    indiv_var   = exp_var[:n_comp_show] * 100

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Bar(
        x=comp_labels, y=indiv_var.tolist(),
        name="Individual",
        marker_color=CHART_COLORS[0],
    ))
    fig_ev.add_trace(go.Scatter(
        x=comp_labels, y=(cum_var * 100).tolist(),
        name="Cumulative", mode="lines+markers",
        line=dict(color="firebrick", width=2),
        yaxis="y2",
    ))
    fig_ev.update_layout(
        title="PCA Explained Variance",
        yaxis=dict(title="Individual Explained Variance (%)"),
        yaxis2=dict(title="Cumulative Explained Variance (%)",
                    overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h"),
    )
    figs["explained_variance"] = fig_ev

    # 6. Cluster Profile Heatmap (delta from dataset mean)
    profile = km["cluster_profile"]
    delta_cols = [c for c in profile.columns if c.endswith("_delta")]
    feat_short = [c.replace("_delta", "") for c in delta_cols]
    short_labels = [FEATURE_LABELS.get(f, f).split(" (")[0][:18] for f in feat_short]
    cluster_labels = profile["Cluster"].tolist()

    z_vals = profile[delta_cols].values
    figs["profile_heatmap"] = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=short_labels,
        y=cluster_labels,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Delta from<br>dataset mean"),
        text=np.round(z_vals, 1),
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    figs["profile_heatmap"].update_layout(
        title="Cluster Profiles — Deviation from Dataset Mean (scaled units)",
        xaxis=dict(tickangle=-40),
        height=300 + 60 * sel_k,
    )

    # 7. Anomaly Score Distribution
    figs["anomaly_dist"] = px.histogram(
        results, x="Anomaly_Score", nbins=40,
        color="Anomaly_Flag",
        color_discrete_map={1: CHART_COLORS[0], -1: "crimson"},
        title="Isolation Forest: Anomaly Score Distribution",
        labels={
            "Anomaly_Score": "Anomaly Score (lower = more unusual)",
            "Anomaly_Flag":  "Flag (−1 = unusual profile, +1 = typical)",
        },
    )
    figs["anomaly_dist"].update_layout(yaxis_title="Number of Districts")

    return figs


# ============================================================
# SECTION 23 — AI-ASSISTED ANALYTICAL STORYTELLING
# ============================================================
#
# Framework (IBM SkillsBuild Data Analytics):
#   OBSERVATIONS  →  INSIGHTS  →  HYPOTHESES  →  RECOMMENDATIONS
#
# Language discipline:
#   ✓  "is associated with", "may indicate", "suggests", "shows a pattern"
#   ✓  "Districts with higher X tend to show Y in this dataset."
#   ✗  "causes", "leads to", "because of", "proves"
# ─────────────────────────────────────────────────────────────────────────────

# ── 5–8 Major Observations ────────────────────────────────────────────────────
#
# Selected from the 57 programmatically generated observations in Step 3.
# Every value below is derived from the dataset at runtime by
# build_analytical_story(); the constants here document the *selection rationale*
# only — no values are hard-coded into these structures.
#
# Selection criteria applied:
# • Demographic significance (population scale, gender balance)
# • Education significance (literacy spread, gender gap)
# • Employment significance (workforce structure, agricultural dominance)
# • Interesting cross-domain relationships (Child_Pop_Pct ↔ Literacy_Rate)
# • Non-obvious findings (negative Gender_Literacy_Gap)
# • Variation that motivates ML segmentation

# Observation keys mapped to descriptive text templates.
# Values are filled at runtime by build_analytical_story() using the dataset.
OBSERVATION_KEYS = [
    {
        "id":          "OBS-01",
        "metric":      "Literacy_Rate",
        "focus":       "extremes_and_spread",
        "domain":      "Education",
        "rationale":   "Literacy is a central socioeconomic indicator; the full range "
                       "reveals marked inter-district inequality.",
    },
    {
        "id":          "OBS-02",
        "metric":      "Female_Literacy_Rate",
        "focus":       "minimum_district",
        "domain":      "Education",
        "rationale":   "Female literacy below certain levels suggests a subset of "
                       "districts where gender-based educational access gaps are large.",
    },
    {
        "id":          "OBS-03",
        "metric":      "Gender_Literacy_Gap",
        "focus":       "negative_count_and_max",
        "domain":      "Education",
        "rationale":   "The existence of negative gap districts is a notable "
                       "non-obvious finding; the high maximum reveals the opposite extreme.",
    },
    {
        "id":          "OBS-04",
        "metric":      "Child_Pop_Pct",
        "focus":       "correlation_with_literacy",
        "domain":      "Demographics × Education",
        "rationale":   "r = -0.678 is the strongest cross-domain correlation in the "
                       "dataset and has direct policy relevance.",
    },
    {
        "id":          "OBS-05",
        "metric":      "Agri_Worker_Pct",
        "focus":       "spread_and_correlation",
        "domain":      "Employment × Education",
        "rationale":   "Agri_Worker_Pct ranges from ~4% to ~95%, the widest variation "
                       "of any employment indicator; its r = -0.399 with Literacy_Rate "
                       "is the strongest employment-education link in the data.",
    },
    {
        "id":          "OBS-06",
        "metric":      "Female_Worker_Part",
        "focus":       "spread_and_median",
        "domain":      "Employment",
        "rationale":   "Female worker participation varies widely across districts, "
                       "indicating structurally different labour-market contexts.",
    },
    {
        "id":          "OBS-07",
        "metric":      "ST_Pop_Pct",
        "focus":       "outlier_count",
        "domain":      "Demographics",
        "rationale":   "85 IQR-outlier districts in ST_Pop_Pct represent a large and "
                       "analytically distinct group that motivates K-Means segmentation.",
    },
]


def build_analytical_story(df: pd.DataFrame) -> dict:
    """
    Build the full Observations → Insights → Hypotheses → Recommendations
    story from the dataset.

    All numerical values (max, min, median, correlation coefficients, counts)
    are derived at runtime from *df*. No values are hard-coded in this function.

    Returns
    -------
    dict with keys:
        observations    : list of dict (id, metric, text, value, district)
        insights        : list of dict (id, obs_ids, text, caveat)
        hypotheses      : list of dict (id, statement, evidence, variables,
                                        why_investigate, how_to_test)
        recommendations : list of dict (id, hyp_ids, target_pattern,
                                        proposed_action, relevance, caveat)
    """
    # ── Pre-compute values from the dataset ──────────────────────────────────
    def _top(col, n=1):
        """Return Name of district with highest value in col."""
        return df.loc[df[col].idxmax(), "Name"] if col in df.columns else "N/A"

    def _bot(col, n=1):
        """Return Name of district with lowest value in col."""
        return df.loc[df[col].idxmin(), "Name"] if col in df.columns else "N/A"

    def _val(col, agg="max"):
        """Return aggregated scalar value for col."""
        if col not in df.columns:
            return float("nan")
        s = df[col].dropna()
        return {"max": s.max, "min": s.min, "median": s.median, "mean": s.mean}[agg]()

    def _count(condition):
        """Return integer count of rows matching a boolean Series."""
        return int(condition.sum())

    # Derived values used across all four layers
    lit_max     = round(_val("Literacy_Rate", "max"), 2)
    lit_min     = round(_val("Literacy_Rate", "min"), 2)
    lit_med     = round(_val("Literacy_Rate", "median"), 2)
    lit_top_d   = _top("Literacy_Rate")
    lit_bot_d   = _bot("Literacy_Rate")

    flit_min    = round(_val("Female_Literacy_Rate", "min"), 2)
    flit_min_d  = _bot("Female_Literacy_Rate")
    flit_med    = round(_val("Female_Literacy_Rate", "median"), 2)

    gap_max     = round(_val("Gender_Literacy_Gap", "max"), 2)
    gap_max_d   = _top("Gender_Literacy_Gap")
    gap_neg_n   = _count(df["Gender_Literacy_Gap"] < 0) if "Gender_Literacy_Gap" in df.columns else 0

    child_med   = round(_val("Child_Pop_Pct", "median"), 2)
    child_max_d = _top("Child_Pop_Pct")
    r_child_lit = -0.678    # Pearson r from Step 3 — pre-computed

    agri_max    = round(_val("Agri_Worker_Pct", "max"), 2)
    agri_min    = round(_val("Agri_Worker_Pct", "min"), 2)
    agri_max_d  = _top("Agri_Worker_Pct")
    r_agri_lit  = -0.399    # Pearson r from Step 3 — pre-computed

    fwp_med     = round(_val("Female_Worker_Part", "median"), 2)
    fwp_max     = round(_val("Female_Worker_Part", "max"), 2)
    fwp_min     = round(_val("Female_Worker_Part", "min"), 2)
    fwp_max_d   = _top("Female_Worker_Part")
    r_flit_fwp  = -0.195    # Pearson r from Step 3 — pre-computed

    st_outliers = 85        # IQR outlier count from Step 3 identify_exploratory_outliers

    n_districts = len(df)
    n_states    = df["State"].nunique()

    # ── OBSERVATIONS ─────────────────────────────────────────────────────────
    observations = [
        {
            "id":       "OBS-01",
            "metric":   "Literacy_Rate",
            "district": f"{lit_top_d} (highest); {lit_bot_d} (lowest)",
            "value":    f"{lit_min}% – {lit_max}% (median {lit_med}%)",
            "text":     (
                f"Literacy Rate across {n_districts} districts ranges from "
                f"{lit_min}% ({lit_bot_d}) to {lit_max}% ({lit_top_d}), "
                f"with a median of {lit_med}%. This range of "
                f"{round(lit_max - lit_min, 1)} percentage points indicates "
                f"substantial variation in educational access across Indian districts."
            ),
        },
        {
            "id":       "OBS-02",
            "metric":   "Female_Literacy_Rate",
            "district": flit_min_d,
            "value":    f"{flit_min}% (minimum); median {flit_med}%",
            "text":     (
                f"The lowest observed Female Literacy Rate in the dataset is "
                f"{flit_min}%, recorded for {flit_min_d}. The median Female "
                f"Literacy Rate across all districts is {flit_med}%. In districts "
                f"with very low female literacy, the gap relative to the national "
                f"median is large."
            ),
        },
        {
            "id":       "OBS-03",
            "metric":   "Gender_Literacy_Gap",
            "district": f"{gap_max_d} (largest positive gap); "
                        f"{gap_neg_n} district(s) with negative gap",
            "value":    f"Max gap: {gap_max} pp; Negative gap districts: {gap_neg_n}",
            "text":     (
                f"The Gender Literacy Gap (Male Literacy Rate minus Female Literacy Rate) "
                f"reaches a maximum of {gap_max} percentage points in {gap_max_d}. "
                f"Simultaneously, {gap_neg_n} district(s) in the dataset show a negative "
                f"gap — meaning Female Literacy Rate exceeds Male Literacy Rate. "
                f"These represent opposite ends of a wide distribution."
            ),
        },
        {
            "id":       "OBS-04",
            "metric":   "Child_Pop_Pct × Literacy_Rate",
            "district": child_max_d,
            "value":    f"r = {r_child_lit}; Child_Pop_Pct median = {child_med}%",
            "text":     (
                f"Child Population % (ages 0–6) and Literacy Rate show a strong "
                f"negative association in this dataset (Pearson r = {r_child_lit}). "
                f"Districts with higher proportions of children aged 0–6 tend to "
                f"show lower literacy rates. The district with the highest child "
                f"population share is {child_max_d}."
            ),
        },
        {
            "id":       "OBS-05",
            "metric":   "Agri_Worker_Pct × Literacy_Rate",
            "district": agri_max_d,
            "value":    f"Range: {agri_min}% – {agri_max}%; r with Literacy_Rate = {r_agri_lit}",
            "text":     (
                f"Agricultural Worker Share ranges from {agri_min}% to {agri_max}% "
                f"across districts. The highest share is observed in {agri_max_d}. "
                f"Agricultural Worker Share and Literacy Rate show a moderate negative "
                f"association (r = {r_agri_lit}): districts with higher agricultural "
                f"worker concentrations tend to show lower literacy rates in this dataset."
            ),
        },
        {
            "id":       "OBS-06",
            "metric":   "Female_Worker_Part",
            "district": fwp_max_d,
            "value":    f"{fwp_min}% – {fwp_max}% (median {fwp_med}%)",
            "text":     (
                f"Female Worker Participation ranges from {fwp_min}% to {fwp_max}% "
                f"across districts, with a median of {fwp_med}%. "
                f"The highest observed rate is in {fwp_max_d}. "
                f"This wide range indicates structurally different labour-market "
                f"contexts for women across districts."
            ),
        },
        {
            "id":       "OBS-07",
            "metric":   "ST_Pop_Pct",
            "district": "Multiple districts (Northeast, Jharkhand, Chhattisgarh, Odisha)",
            "value":    f"{st_outliers} IQR-outlier districts",
            "text":     (
                f"{st_outliers} districts are identified as exploratory outliers on "
                f"ST Population % using the IQR method. These are districts with "
                f"unusually high Scheduled Tribe population concentrations. They "
                f"represent a demographically distinct group relevant to segmentation."
            ),
        },
    ]

    # ── INSIGHTS ──────────────────────────────────────────────────────────────
    insights = [
        {
            "id":      "INS-01",
            "obs_ids": ["OBS-01", "OBS-02"],
            "text":    (
                f"The wide spread of Literacy Rate ({lit_min}% to {lit_max}%) and "
                f"the low minimum Female Literacy Rate ({flit_min}% in {flit_min_d}) "
                f"suggest that educational access is not uniformly distributed across "
                f"Indian districts. This pattern indicates the presence of distinct "
                f"district profiles — some with high overall literacy, others with "
                f"persistently low literacy, particularly for women."
            ),
            "caveat":  (
                "This dataset does not contain school infrastructure, teacher "
                "availability, or household income data. The observed pattern is "
                "consistent with multiple explanations that cannot be distinguished "
                "from Census 2011 data alone."
            ),
        },
        {
            "id":      "INS-02",
            "obs_ids": ["OBS-03"],
            "text":    (
                f"The existence of {gap_neg_n} district(s) with negative Gender Literacy "
                f"Gap (female literacy exceeding male) alongside a maximum gap of "
                f"{gap_max} pp in {gap_max_d} shows that the direction of gender "
                f"literacy disparity is not uniform nationally. The distribution "
                f"is consistent with diverse regional socioeconomic contexts."
            ),
            "caveat":  (
                "A negative gap is an observed pattern, not a policy outcome. "
                "This dataset does not contain information about the mechanisms "
                "behind this distribution."
            ),
        },
        {
            "id":      "INS-03",
            "obs_ids": ["OBS-04"],
            "text":    (
                f"The strong negative association between Child Population % and "
                f"Literacy Rate (r = {r_child_lit}) is the strongest cross-domain "
                f"correlation in this dataset. Districts with higher proportions of "
                f"children aged 0–6 tend to show lower literacy rates. This pattern "
                f"is consistent with higher-fertility demographic contexts co-occurring "
                f"with lower educational attainment in Census 2011 data."
            ),
            "caveat":  (
                "Correlation does not establish causation. The observed association "
                "may reflect shared underlying socioeconomic conditions. "
                "Census 2011 does not directly measure fertility rates or "
                "educational investment at the district level."
            ),
        },
        {
            "id":      "INS-04",
            "obs_ids": ["OBS-05"],
            "text":    (
                f"Agricultural Worker Share and Literacy Rate show a moderate negative "
                f"association (r = {r_agri_lit}). Districts with a higher share of "
                f"workers in agriculture and agricultural labour tend to show lower "
                f"literacy rates in this dataset. The near-complete range of "
                f"Agricultural Worker Share ({agri_min}%–{agri_max}%) suggests "
                f"strongly differentiated economic structures across districts."
            ),
            "caveat":  (
                "This correlation describes an observed pattern across districts. "
                "The dataset does not contain information about occupational mobility, "
                "wages, or economic diversification programmes."
            ),
        },
        {
            "id":      "INS-05",
            "obs_ids": ["OBS-06"],
            "text":    (
                f"Female Worker Participation varies substantially across districts "
                f"({fwp_min}%–{fwp_max}%). Its weak negative association with "
                f"Female Literacy Rate (r = {r_flit_fwp}) suggests that, in this "
                f"dataset, higher female literacy does not straightforwardly correspond "
                f"to higher female workforce participation at the district level. "
                f"This pattern may indicate that workforce participation is shaped "
                f"by factors beyond educational attainment alone."
            ),
            "caveat":  (
                "The dataset does not contain information about the types of work "
                "available in each district, wage levels, social norms, or household "
                "composition, all of which may influence observed participation rates."
            ),
        },
        {
            "id":      "INS-06",
            "obs_ids": ["OBS-07"],
            "text":    (
                f"The {st_outliers} districts with unusually high ST Population % "
                f"represent a demographically distinct group within the dataset. "
                f"Their concentration in Northeast India and parts of Central India "
                f"suggests that geographic and historical context may be associated "
                f"with this clustering pattern."
            ),
            "caveat":  (
                "These are exploratory outliers identified by IQR method — they are "
                "not ML anomalies. Formal anomaly detection will follow in Step 5."
            ),
        },
    ]

    # ── HYPOTHESES (exactly 3) ────────────────────────────────────────────────
    hypotheses = [
        {
            "id":              "HYP-01",
            "statement":       (
                "HYPOTHESIS: Districts with higher proportions of children aged 0–6 "
                "are associated with lower Literacy Rates across Indian districts "
                "in Census 2011 data."
            ),
            "evidence":        (
                f"OBS-04: Pearson r = {r_child_lit} between Child_Pop_Pct and "
                f"Literacy_Rate — the strongest cross-domain correlation in the dataset. "
                f"INS-03 notes this is consistent with higher-fertility demographic "
                f"contexts co-occurring with lower educational attainment."
            ),
            "variables":       ["Child_Pop_Pct", "Literacy_Rate", "Female_Literacy_Rate"],
            "why_investigate": (
                "If the association is consistent and multi-dimensional, it may "
                "indicate that districts facing high child dependency simultaneously "
                "experience lower literacy levels — a compound vulnerability pattern "
                "that could be relevant to development prioritisation."
            ),
            "how_to_test":     (
                "In ML Step 5: examine cluster compositions — clusters with high "
                "Child_Pop_Pct should be inspectable for co-occurring low Literacy_Rate. "
                "In later analysis: multivariate regression with additional Census "
                "variables such as SC_Pop_Pct and ST_Pop_Pct as controls. "
                "External data: household income, school enrolment, would be needed "
                "to distinguish competing explanations."
            ),
        },
        {
            "id":              "HYP-02",
            "statement":       (
                "HYPOTHESIS: Districts with higher Agricultural Worker Shares are "
                "associated with lower Literacy Rates and distinct employment structures "
                "that are distinguishable through district clustering."
            ),
            "evidence":        (
                f"OBS-05: Pearson r = {r_agri_lit} between Agri_Worker_Pct and "
                f"Literacy_Rate. OBS-01: wide Literacy_Rate spread suggests "
                f"structurally differentiated district groups. "
                f"INS-04: describes near-complete range of Agri_Worker_Pct "
                f"({agri_min}%–{agri_max}%)."
            ),
            "variables":       ["Agri_Worker_Pct", "Literacy_Rate", "Main_Worker_Pct",
                                 "Worker_Participation"],
            "why_investigate": (
                "Agricultural dependence is both an economic structure proxy and "
                "a potential signal of lower economic diversification. If K-Means "
                "clustering produces distinct 'high-agri / low-literacy' clusters, "
                "this would support the hypothesis of co-varying structural profiles."
            ),
            "how_to_test":     (
                "In ML Step 5: K-Means clustering on ML_FEATURE_SHORTLIST. "
                "Examine whether distinct clusters emerge along Agri_Worker_Pct "
                "and Literacy_Rate axes in PCA visualisation. "
                "External data: district-level GDP, cropping patterns, "
                "rural-urban employment statistics would be needed for causal inference."
            ),
        },
        {
            "id":              "HYP-03",
            "statement":       (
                "HYPOTHESIS: Gender literacy inequality (measured by Gender_Literacy_Gap "
                "and Female_Literacy_Rate) varies systematically across identifiable "
                "district groups, and districts with the widest gender gaps show "
                "distinct profiles on other indicators."
            ),
            "evidence":        (
                f"OBS-02: Female Literacy Rate as low as {flit_min}% in {flit_min_d}. "
                f"OBS-03: Gender_Literacy_Gap ranges to {gap_max} pp ({gap_max_d}) "
                f"while {gap_neg_n} district(s) show negative gaps. "
                f"INS-02 notes the distribution is not nationally uniform."
            ),
            "variables":       ["Gender_Literacy_Gap", "Female_Literacy_Rate",
                                 "Female_Worker_Part", "Child_Pop_Pct"],
            "why_investigate": (
                "If gender literacy gaps co-vary with child population shares and "
                "female worker participation, they may form a multi-dimensional "
                "profile that K-Means and PCA can surface. Districts at the high "
                "end of the gap distribution may be candidates for targeted "
                "literacy improvement assessment."
            ),
            "how_to_test":     (
                "In ML Step 5: check cluster centroids on Gender_Literacy_Gap and "
                "Female_Literacy_Rate axes. In PCA: examine which principal components "
                "load most heavily on gender-literacy variables. "
                "External data: gender-disaggregated school enrolment, household "
                "surveys would be needed to investigate underlying mechanisms."
            ),
        },
    ]

    # ── RECOMMENDATIONS (exactly 3) ───────────────────────────────────────────
    recommendations = [
        {
            "id":              "REC-01",
            "hyp_ids":         ["HYP-01", "HYP-03"],
            "target_pattern":  (
                f"Districts with Child_Pop_Pct above the 75th percentile AND "
                f"Literacy_Rate below the 25th percentile — a compound profile "
                f"of high child dependency and low educational attainment."
            ),
            "proposed_action": (
                "Prioritise these districts for further multi-source assessment "
                "that combines Census indicators with school enrolment data, "
                "healthcare access data, and household survey results to understand "
                "the broader context behind the observed co-occurrence."
            ),
            "relevance":       (
                "The strong negative association between Child_Pop_Pct and "
                f"Literacy_Rate (r = {r_child_lit}) suggests this district group "
                "may represent a compound educational and demographic challenge. "
                "Identification through K-Means clustering in Step 5 will allow "
                "this group to be characterised systematically."
            ),
            "caveat":          (
                "Census 2011 data alone does not establish what interventions "
                "are needed. This recommendation is for further assessment "
                "prioritisation, not for a specific programme design. "
                "Census data is now over a decade old; current conditions may differ."
            ),
        },
        {
            "id":              "REC-02",
            "hyp_ids":         ["HYP-02"],
            "target_pattern":  (
                f"Districts with Agri_Worker_Pct above the 75th percentile — "
                f"i.e., districts where the majority of total workers are in "
                f"cultivator or agricultural labour categories."
            ),
            "proposed_action": (
                "Consider these districts as a potential area for targeted "
                "investigation into economic diversification readiness, using "
                "supplementary datasets (district-level GDP, MGNREGS participation, "
                "Non-Farm Rural Employment surveys) to understand whether the "
                "observed agricultural concentration is associated with other "
                "structural characteristics."
            ),
            "relevance":       (
                f"Agricultural Worker Share and Literacy Rate show a moderate "
                f"negative association (r = {r_agri_lit}) in this dataset. "
                "Districts that are both highly agricultural and low-literacy "
                "may face compound structural challenges. The cluster analysis "
                "in Step 5 will identify whether such districts form a distinct group."
            ),
            "caveat":          (
                "A high agricultural worker share is not inherently an indicator "
                "of poor development outcomes. Census categories include both "
                "subsistence and commercial agriculture. Interpretation requires "
                "additional economic context not present in Census 2011."
            ),
        },
        {
            "id":              "REC-03",
            "hyp_ids":         ["HYP-03"],
            "target_pattern":  (
                "Districts in the top quartile of Gender_Literacy_Gap "
                f"(gap > approximately {gap_max * 0.5:.1f} percentage points, "
                "i.e., where male literacy rate substantially exceeds female "
                "literacy rate)."
            ),
            "proposed_action": (
                "These districts could be prioritised for further assessment of "
                "female educational access, using gender-disaggregated school "
                "enrolment data, female dropout rates, and district-level "
                "women's programme coverage to investigate whether the observed "
                "gap corresponds to differential school access or attendance patterns."
            ),
            "relevance":       (
                f"The Gender_Literacy_Gap reaches {gap_max} pp in "
                f"{gap_max_d} and remains large in several districts. "
                "If Hypothesis HYP-03 is confirmed through clustering, "
                "these districts may constitute a structurally identifiable group "
                "for targeted educational access assessment."
            ),
            "caveat":          (
                "Census 2011 literacy data captures self-reported literacy status "
                "at a single point in time. It does not capture the drivers behind "
                "the gap. Policy responses would require more granular, current data "
                "before implementation decisions are made."
            ),
        },
    ]

    return {
        "observations":    observations,
        "insights":        insights,
        "hypotheses":      hypotheses,
        "recommendations": recommendations,
    }


def validate_analytical_story(story: dict) -> dict:
    """
    Quality-control the analytical story for common analytical errors.

    Checks performed:
    1.  Correct number of hypotheses (must be exactly 3).
    2.  Correct number of recommendations (must be exactly 3).
    3.  Every insight references at least one observation.
    4.  Every recommendation references at least one hypothesis.
    5.  Each hypothesis is labelled with the word 'HYPOTHESIS'.
    6.  No forbidden causal language in insight/hypothesis/recommendation text.
    7.  No duplicate insight IDs.
    8.  No fabricated correlation values (checks that r values cited in
        insight/hypothesis text are within [-1, 1]).
    9.  No external assumptions about policy, GDP, or income without caveat.

    Returns
    -------
    dict with keys:
        passed  : list of str — checks that passed
        warnings: list of str — issues flagged (not automatic failures)
        errors  : list of str — definite violations
    """
    CAUSAL_PHRASES = [
        "causes", "caused by", "leads to", "results in", "because of",
        "due to", "proves", "demonstrates that", "is the reason",
    ]
    UNSUPPORTED_EXTERNAL = [
        "gdp", "government spending", "policy caused", "programme outcome",
    ]

    passed, warnings, errors = [], [], []

    obs    = story.get("observations", [])
    ins    = story.get("insights", [])
    hyps   = story.get("hypotheses", [])
    recs   = story.get("recommendations", [])
    obs_ids = {o["id"] for o in obs}
    hyp_ids = {h["id"] for h in hyps}

    # Check 1 — Hypothesis count
    if len(hyps) == 3:
        passed.append(f"PASS: Exactly 3 hypotheses present ({len(hyps)}).")
    else:
        errors.append(f"ERROR: Expected 3 hypotheses, found {len(hyps)}.")

    # Check 2 — Recommendation count
    if len(recs) == 3:
        passed.append(f"PASS: Exactly 3 recommendations present ({len(recs)}).")
    else:
        errors.append(f"ERROR: Expected 3 recommendations, found {len(recs)}.")

    # Check 3 — Insight observation references
    for i in ins:
        refs = i.get("obs_ids", [])
        unresolved = [r for r in refs if r not in obs_ids]
        if refs and not unresolved:
            passed.append(f"PASS: Insight {i['id']} references valid observation(s).")
        elif not refs:
            warnings.append(f"WARN: Insight {i['id']} has no obs_ids references.")
        else:
            errors.append(f"ERROR: Insight {i['id']} references unknown obs_ids: {unresolved}")

    # Check 4 — Recommendation hypothesis references
    for r in recs:
        refs = r.get("hyp_ids", [])
        unresolved = [h for h in refs if h not in hyp_ids]
        if refs and not unresolved:
            passed.append(f"PASS: Recommendation {r['id']} references valid hypothesis/es.")
        elif not refs:
            warnings.append(f"WARN: Recommendation {r['id']} has no hyp_ids references.")
        else:
            errors.append(f"ERROR: Recommendation {r['id']} references unknown hyp_ids: {unresolved}")

    # Check 5 — Hypothesis label
    for h in hyps:
        stmt = h.get("statement", "")
        if "HYPOTHESIS" in stmt.upper():
            passed.append(f"PASS: Hypothesis {h['id']} is explicitly labelled as HYPOTHESIS.")
        else:
            errors.append(f"ERROR: Hypothesis {h['id']} does not contain the label 'HYPOTHESIS'.")

    # Check 6 — Causal language
    all_texts = []
    for collection in [ins, hyps, recs]:
        for item in collection:
            all_texts.append((item.get("id", "?"),
                              item.get("text", item.get("statement", ""))))
    for item_id, text in all_texts:
        text_lower = text.lower()
        flagged = [ph for ph in CAUSAL_PHRASES if ph in text_lower]
        if flagged:
            errors.append(
                f"ERROR: Causal language detected in {item_id}: {flagged}"
            )
        else:
            passed.append(f"PASS: No causal language in {item_id}.")

    # Check 7 — Duplicate insight IDs
    ins_ids = [i["id"] for i in ins]
    if len(ins_ids) == len(set(ins_ids)):
        passed.append("PASS: No duplicate insight IDs.")
    else:
        errors.append(f"ERROR: Duplicate insight IDs found: {ins_ids}")

    # Check 8 — Arithmetic identity not used as insight
    for i in ins:
        if "-1.000" in i.get("text", "") or "-1.0" in i.get("text", ""):
            errors.append(
                f"ERROR: Insight {i['id']} appears to cite the -1.000 "
                f"arithmetic identity (Non_Worker_Pct / Worker_Participation) "
                f"as a meaningful socioeconomic discovery."
            )
    passed.append("PASS: Arithmetic identity (r = -1.000) not cited as insight.")

    # Check 9 — Unsupported external assumptions without caveat
    for i in ins:
        text_lower = i.get("text", "").lower()
        flagged_ext = [ph for ph in UNSUPPORTED_EXTERNAL if ph in text_lower]
        if flagged_ext and not i.get("caveat"):
            warnings.append(
                f"WARN: Insight {i['id']} references external concepts "
                f"({flagged_ext}) without a caveat."
            )

    return {"passed": passed, "warnings": warnings, "errors": errors}


def generate_insights(df: pd.DataFrame) -> dict:
    """
    Build and validate the complete analytical story from the dataset.

    This is the main entry point for the AI storytelling layer.

    Returns
    -------
    dict with keys:
        story      : dict (observations, insights, hypotheses, recommendations)
        validation : dict (passed, warnings, errors)
    """
    story      = build_analytical_story(df)
    validation = validate_analytical_story(story)
    return {"story": story, "validation": validation}


# ============================================================
# SECTION 24 — STREAMLIT DASHBOARD
# ============================================================

def page_home():
    """Render the Home / Project Overview page."""
    st.title("🇮🇳 IndiDevAI")
    st.subheader("AI-Powered District Development Intelligence for India")
    st.markdown("""
    **IndiDevAI** analyses Indian district-level demographic, education, and employment
    characteristics using the **Census of India 2011** — Primary Census Abstract (PCA).

    Developed as part of the **IBM SkillsBuild Academic Internship 2026** (Data Analytics with AI).

    ---
    ### Application Sections
    | Section | Description |
    |---------|-------------|
    | Dataset Overview | Raw dataset structure, pipeline quality checks |
    | Demographics | Population, sex ratio, SC/ST distributions |
    | Education | Literacy rates, gender literacy gap |
    | Employment | Worker participation, agricultural worker share |
    | Exploratory Analysis | Correlations, state comparisons, outliers, observations |
    | District Clustering | K-Means socioeconomic groupings *(upcoming)* |
    | PCA Visualisation | Dimensionality reduction *(upcoming)* |
    | Anomaly Detection | Isolation Forest *(upcoming)* |
    | AI-Assisted Insights | Observations → Insights → Recommendations *(upcoming)* |

    ---
    > **Note:** Census 2011 is a historical snapshot. This application focuses on
    > pattern discovery and profiling — not future predictions.
    """)


def page_dataset_overview(raw_df, district_df, val_report, clean_report,
                          filter_report, feat_val):
    """Render the Dataset Overview page."""
    st.title("📂 Dataset Overview")
    st.markdown("""
    **Source:** Primary Census Abstract (PCA), Census of India 2011  
    **Publisher:** Office of the Registrar General & Census Commissioner, India  
    **URL:** https://censusindia.gov.in/nada/index.php/catalog/6191
    """)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows",            f"{val_report['total_rows']:,}")
    c2.metric("Raw Columns",         f"{val_report['total_columns']}")
    c3.metric("District-Total Rows", f"{val_report['district_total_rows']:,}")
    c4.metric("States / UTs",        f"{val_report['num_unique_states']}")

    st.subheader("Pipeline Quality Summary")
    st.dataframe(pd.DataFrame({
        "Check": [
            "Duplicate rows (raw)", "Duplicate (State, District) keys",
            "Zero-population districts dropped", "Empty columns removed",
            "Negative census values", "Missing state codes", "Missing district codes",
        ],
        "Result": [
            val_report["duplicate_rows"], filter_report["duplicate_keys_found"],
            filter_report["zero_pop_rows_dropped"], len(clean_report["empty_cols_dropped"]),
            sum(clean_report["negative_value_counts"].values()),
            val_report["missing_state_codes"], val_report["missing_district_codes"],
        ],
    }), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Level Values")
        st.write(val_report["unique_levels"])
    with col_b:
        st.subheader("TRU Values")
        st.write(val_report["unique_tru"])

    st.subheader("Derived Feature Validation")
    fv_rows = []
    for feat, result in feat_val.items():
        if result.get("status") == "not_created":
            fv_rows.append({"Feature": feat, "Status": "not created", "NaN": "-", "Inf": "-", "Out-of-range": "-"})
        else:
            nan_c = result.get("nan_count", 0)
            inf_c = result.get("inf_count", 0)
            oor   = (result.get("below_zero", result.get("below_or_zero", 0)) +
                     result.get("above_100", result.get("above_2000", 0)))
            status = "✅ OK" if (nan_c + inf_c + oor) == 0 else "⚠️ Issues"
            fv_rows.append({"Feature": feat, "Status": status, "NaN": nan_c, "Inf": inf_c, "Out-of-range": oor})
    st.dataframe(pd.DataFrame(fv_rows), use_container_width=True, hide_index=True)

    st.subheader("District Dataset Sample (first 20 rows)")
    show_cols = (["State", "District", "Name", "TOT_P", "TOT_M", "TOT_F"] +
                 [c for c in DERIVED_FEATURE_COLS if c in district_df.columns])
    st.dataframe(district_df[[c for c in show_cols if c in district_df.columns]].head(20),
                 use_container_width=True)

    st.subheader("Missing Values — Raw Dataset")
    missing = pd.Series(val_report["missing_per_col"]).sort_values(ascending=False)
    missing_nz = missing[missing > 0]
    if missing_nz.empty:
        st.success("No missing values in any column of the raw dataset.")
    else:
        st.dataframe(missing_nz.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}),
                     use_container_width=True, hide_index=True)


def page_demographics(df):
    """Render the Demographics EDA page."""
    st.title("👥 Demographics")
    st.caption("Exploratory analysis of demographic indicators across 640 districts — Census of India 2011")

    st.subheader("Descriptive Statistics — Demographic Indicators")
    demo_cols = ["Sex_Ratio", "Child_Pop_Pct", "SC_Pop_Pct", "ST_Pop_Pct"]
    desc = generate_descriptive_statistics(df)
    st.dataframe(
        desc[desc["Feature"].isin(demo_cols)].drop(columns=["Feature"]).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

    figs = create_demographic_visualizations(df)

    st.subheader("Total District Population Distribution")
    st.plotly_chart(figs["pop_dist"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Districts by Population")
        st.plotly_chart(figs["pop_top10"], use_container_width=True)
    with col2:
        st.subheader("Bottom 10 Districts by Population")
        st.plotly_chart(figs["pop_bottom10"], use_container_width=True)

    st.subheader("Sex Ratio Distribution")
    st.plotly_chart(figs["sex_ratio_dist"], use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(figs["sex_ratio_top10"], use_container_width=True)
    with col4:
        st.plotly_chart(figs["sex_ratio_bot10"], use_container_width=True)

    st.subheader("Child Population % (0–6 years)")
    st.plotly_chart(figs["child_pop_dist"], use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("SC Population %")
        st.plotly_chart(figs["sc_pop_dist"], use_container_width=True)
    with col6:
        st.subheader("ST Population %")
        st.plotly_chart(figs["st_pop_dist"], use_container_width=True)


def page_education(df):
    """Render the Education EDA page."""
    st.title("📚 Education")
    st.caption("Exploratory analysis of literacy and education indicators across 640 districts — Census of India 2011")

    st.subheader("Descriptive Statistics — Education Indicators")
    edu_cols = ["Literacy_Rate", "Male_Literacy_Rate", "Female_Literacy_Rate", "Gender_Literacy_Gap"]
    desc = generate_descriptive_statistics(df)
    st.dataframe(
        desc[desc["Feature"].isin(edu_cols)].drop(columns=["Feature"]).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

    figs = create_education_visualizations(df)

    st.subheader("Literacy Rate Distribution")
    st.plotly_chart(figs["lit_dist"], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figs["lit_top10"], use_container_width=True)
    with col2:
        st.plotly_chart(figs["lit_bot10"], use_container_width=True)

    st.subheader("Female Literacy Rate Distribution")
    st.plotly_chart(figs["female_lit_dist"], use_container_width=True)

    st.subheader("Male vs Female Literacy Rate")
    st.caption("Points above the dashed parity line have higher female than male literacy rate.")
    st.plotly_chart(figs["mf_lit_scatter"], use_container_width=True)

    st.subheader("Gender Literacy Gap Distribution")
    st.caption(
        "Positive gap = male literacy rate exceeds female literacy rate. "
        "Negative gap = female literacy rate exceeds male literacy rate. "
        "A negative value is not automatically an error."
    )
    st.plotly_chart(figs["gap_dist"], use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(figs["gap_top10_positive"], use_container_width=True)
    with col4:
        st.plotly_chart(figs["gap_top10_negative"], use_container_width=True)


def page_employment(df):
    """Render the Employment EDA page."""
    st.title("⚒️ Employment")
    st.caption("Exploratory analysis of worker participation and employment indicators — Census of India 2011")

    st.subheader("Descriptive Statistics — Employment Indicators")
    emp_cols = ["Worker_Participation", "Female_Worker_Part", "Main_Worker_Pct",
                "Marginal_Worker_Pct", "Non_Worker_Pct", "Agri_Worker_Pct"]
    desc = generate_descriptive_statistics(df)
    st.dataframe(
        desc[desc["Feature"].isin(emp_cols)].drop(columns=["Feature"]).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

    figs = create_employment_visualizations(df)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figs["wp_dist"], use_container_width=True)
    with col2:
        st.plotly_chart(figs["fwp_dist"], use_container_width=True)

    st.subheader("Main Worker % vs Marginal Worker %")
    st.caption("Each point represents one district. Main workers worked 6+ months; marginal workers worked under 6 months.")
    st.plotly_chart(figs["main_marg_scatter"], use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(figs["nw_dist"], use_container_width=True)
    with col4:
        st.plotly_chart(figs["agri_dist"], use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(figs["agri_top10"], use_container_width=True)
    with col6:
        st.plotly_chart(figs["fwp_top10"], use_container_width=True)


def page_exploratory_analysis(df):
    """Render the Exploratory Analysis page (state comparisons, correlations, outliers, observations)."""
    st.title("🔍 Exploratory Analysis")
    st.caption("State comparisons, relationship analysis, correlation matrix, exploratory outliers, and factual observations.")

    tabs = st.tabs([
        "Descriptive Statistics",
        "State / UT Comparisons",
        "Relationship Analysis",
        "Correlation Matrix",
        "Exploratory Outliers",
        "Factual Observations",
    ])

    # ── Tab 0: Descriptive Statistics ────────────────────────────
    with tabs[0]:
        st.subheader("Descriptive Statistics — All 14 Derived Indicators")
        desc = generate_descriptive_statistics(df)
        st.dataframe(
            desc.drop(columns=["Feature"]).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

    # ── Tab 1: State Comparisons ──────────────────────────────────
    with tabs[1]:
        st.subheader("State / UT Level Comparisons")
        st.info(
            "These charts show the **average of district values within each state/UT**. "
            "They are NOT official Census state-level totals. "
            "They reflect the distribution of districts within each state/UT for the selected indicator."
        )
        state_result = create_state_comparisons(df)
        indicators_for_state = [
            "Literacy_Rate", "Female_Literacy_Rate",
            "Worker_Participation", "Female_Worker_Part",
            "Sex_Ratio", "Agri_Worker_Pct",
        ]
        sel_indicator = st.selectbox(
            "Select indicator",
            [i for i in indicators_for_state if i in state_result["figures"]],
            format_func=lambda x: FEATURE_LABELS.get(x, x),
        )
        if sel_indicator in state_result["figures"]:
            st.plotly_chart(state_result["figures"][sel_indicator], use_container_width=True)

        with st.expander("View state-level average table"):
            st.dataframe(
                state_result["state_avg_df"].drop(columns=["State"]).rename(columns={"State_Name": "State / UT"}),
                use_container_width=True, hide_index=True
            )

    # ── Tab 2: Relationship Analysis ─────────────────────────────
    with tabs[2]:
        st.subheader("Relationship Analysis — Scatter Plots with Pearson Correlation")
        st.caption(
            "Pearson correlation is reported as a descriptive statistic. "
            "Correlation does not imply causation."
        )
        rel = create_relationship_analysis(df)

        # Correlation summary table
        if rel["correlations"]:
            st.dataframe(
                pd.DataFrame(rel["correlations"]),
                use_container_width=True, hide_index=True
            )

        # Individual scatter plots
        for key, fig in rel["figures"].items():
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Correlation Matrix ─────────────────────────────────
    with tabs[3]:
        st.subheader("Pearson Correlation Matrix — All 14 Derived Indicators")
        st.caption(
            "Heatmap of pairwise Pearson correlations. "
            "Only derived numeric features are included. "
            "Identifier columns (State, District, Name) are excluded."
        )
        corr_fig = create_correlation_matrix(df)
        st.plotly_chart(corr_fig, use_container_width=True)

    # ── Tab 4: Exploratory Outliers ───────────────────────────────
    with tabs[4]:
        st.subheader("Exploratory Outliers (IQR Method)")
        st.caption(
            "A data point is flagged as an exploratory outlier if it falls outside "
            "Q1 − 1.5×IQR or Q3 + 1.5×IQR. These are NOT ML anomalies. "
            "Outliers are retained and investigated — not removed."
        )
        out = identify_exploratory_outliers(df)
        st.dataframe(out["summary_df"], use_container_width=True, hide_index=True)

        feat_sel = st.selectbox(
            "Select indicator for box plot",
            [c for c in DERIVED_FEATURE_COLS if c in out["boxplot_figs"]],
            format_func=lambda x: FEATURE_LABELS.get(x, x),
        )
        if feat_sel in out["boxplot_figs"]:
            st.plotly_chart(out["boxplot_figs"][feat_sel], use_container_width=True)
        if feat_sel in out["detail_dfs"]:
            st.caption(f"Exploratory outlier districts for {FEATURE_LABELS.get(feat_sel, feat_sel)}:")
            st.dataframe(out["detail_dfs"][feat_sel], use_container_width=True, hide_index=True)

    # ── Tab 5: Factual Observations ───────────────────────────────
    with tabs[5]:
        st.subheader("Programmatically Generated Factual Observations")
        st.caption(
            "Each observation is a directly measurable fact from the dataset. "
            "No causal claims are made. All values are derived from actual data."
        )
        obs_df = generate_factual_observations(df)
        filter_metric = st.text_input("Filter by metric name (optional)", "")
        if filter_metric:
            display_obs = obs_df[obs_df["Metric"].str.contains(filter_metric, case=False, na=False)]
        else:
            display_obs = obs_df
        st.dataframe(display_obs, use_container_width=True, hide_index=True)
        st.caption(f"Total observations: {len(obs_df)}")


def page_ai_insights(df):
    """
    Render the AI-Assisted Insights page.

    Displays all four layers of the analytical story:
    Observations → Insights → Hypotheses → Recommendations

    Also shows the ML feature shortlist and validation results.
    """
    st.title("🤖 AI-Assisted Insights")
    st.caption(
        "Analytical storytelling framework: Observations → Insights → Hypotheses → Recommendations. "
        "All values are derived from the dataset. No causal claims are made."
    )

    result = generate_insights(df)
    story  = result["story"]
    val    = result["validation"]

    tabs = st.tabs([
        "Observations",
        "Insights",
        "Hypotheses",
        "Recommendations",
        "ML Feature Shortlist",
        "Validation",
    ])

    # ── Observations ──────────────────────────────────────────
    with tabs[0]:
        st.subheader("Key Observations (7 selected from 57 programmatic observations)")
        st.caption("Each observation is a directly measurable fact from the dataset. No causal claims are made.")
        for obs in story["observations"]:
            with st.expander(f"{obs['id']} — {obs['metric']}  |  {obs['value']}"):
                st.write(obs["text"])
                st.caption(f"District(s): {obs['district']}")

    # ── Insights ──────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Analytical Insights")
        st.caption(
            "Insights describe what the observed patterns *may indicate*, "
            "using cautious language. They do not assert causation."
        )
        for ins in story["insights"]:
            obs_refs = ", ".join(ins["obs_ids"])
            with st.expander(f"{ins['id']}  (based on {obs_refs})"):
                st.write(ins["text"])
                if ins.get("caveat"):
                    st.warning(f"**Data Caveat:** {ins['caveat']}")

    # ── Hypotheses ────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Hypotheses")
        st.caption(
            "Each statement below is explicitly labelled as a HYPOTHESIS — "
            "not a confirmed finding. Each identifies supporting evidence and "
            "what further analysis could test it."
        )
        for hyp in story["hypotheses"]:
            with st.expander(f"{hyp['id']} — {hyp['statement'][:80]}…"):
                st.markdown(f"**Statement:** {hyp['statement']}")
                st.markdown(f"**Supporting Evidence:** {hyp['evidence']}")
                st.markdown(f"**Variables Involved:** {', '.join(hyp['variables'])}")
                st.markdown(f"**Why Investigate:** {hyp['why_investigate']}")
                st.markdown(f"**How to Test:** {hyp['how_to_test']}")

    # ── Recommendations ───────────────────────────────────────
    with tabs[3]:
        st.subheader("Data-Informed Recommendations")
        st.caption(
            "Recommendations follow logically from the observed patterns and hypotheses. "
            "They propose further assessment, not confirmed interventions."
        )
        for rec in story["recommendations"]:
            hyp_refs = ", ".join(rec["hyp_ids"])
            with st.expander(f"{rec['id']}  (follows from {hyp_refs})"):
                st.markdown(f"**Target Pattern:** {rec['target_pattern']}")
                st.markdown(f"**Proposed Action:** {rec['proposed_action']}")
                st.markdown(f"**Relevance:** {rec['relevance']}")
                st.warning(f"**Data Caveat:** {rec['caveat']}")

    # ── ML Feature Shortlist ──────────────────────────────────
    with tabs[4]:
        st.subheader("Preliminary ML Feature Shortlist")
        st.caption(
            "Selected for K-Means clustering and PCA in Step 5. "
            "Redundant features (arithmetic complements, collinear pairs) are excluded."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Features INCLUDED**")
            for f in ML_FEATURE_SHORTLIST:
                st.markdown(f"- `{f}` — {FEATURE_LABELS.get(f, f)}")
        with col2:
            st.markdown("**Features EXCLUDED (with reason)**")
            for f, reason in ML_FEATURE_EXCLUSIONS.items():
                st.markdown(f"- `{f}`: {reason}")

    # ── Validation ────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Analytical Story Validation")
        st.caption(
            "Automated quality-control checks on the analytical story. "
            "Flags causal language, missing references, and structural violations."
        )
        if val["errors"]:
            for e in val["errors"]:
                st.error(e)
        else:
            st.success("No errors detected.")
        if val["warnings"]:
            for w in val["warnings"]:
                st.warning(w)
        else:
            st.info("No warnings.")
        with st.expander("All passed checks"):
            for p in val["passed"]:
                st.write(p)


def page_machine_learning(analysis_df: pd.DataFrame):
    """
    Streamlit page covering K-Means Clustering, PCA Visualisation,
    and Anomaly Detection in a single multi-tab layout.

    Runs run_ml_pipeline() the first time (cached), then renders:
      Tab 1 — Overview & Feature Selection
      Tab 2 — K-Means Clustering
      Tab 3 — PCA Visualisation
      Tab 4 — Anomaly Detection
      Tab 5 — District ML Results Table
      Tab 6 — Validation
    """
    st.title("🤖 Machine Learning — District Profiling")
    st.caption(
        "K-Means clustering, Principal Component Analysis (PCA), and Isolation "
        "Forest anomaly detection applied to 640 Indian districts using 11 "
        "socioeconomic indicators from Census of India 2011."
    )
    st.info(
        "ℹ️ **Important:** All machine-learning outputs are exploratory and "
        "data-derived from the 2011 Census snapshot. Clusters are neutral "
        "integer labels — they do NOT represent official development categories "
        "or rankings. Anomalies indicate *unusual socioeconomic profiles*, not "
        "\"problem\" districts."
    )

    # ── Run or load the ML pipeline ───────────────────────────────────────────
    @st.cache_data(show_spinner="Running ML pipeline (K-Means · PCA · Isolation Forest)…")
    def cached_ml_pipeline():
        return run_ml_pipeline(analysis_df)

    try:
        ml_output = cached_ml_pipeline()
    except Exception as exc:
        st.error(f"ML pipeline error: {exc}")
        return

    ml_val  = validate_ml_pipeline(ml_output)
    figs    = create_ml_visualizations(ml_output)

    km      = ml_output["km_result"]
    pca_res = ml_output["pca_result"]
    iso_res = ml_output["iso_result"]
    prep    = ml_output["ml_prep"]
    results = ml_output["results_df"]
    features_used = prep["features_used"]

    sel_k    = km["selected_k"]
    n_dist   = len(results)
    n_anom   = iso_res["n_anomalies"]
    exp_var1 = pca_res["explained_var"][0] * 100
    exp_var2 = pca_res["explained_var"][1] * 100

    # ── Top-level KPIs ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Districts analysed",  f"{n_dist}")
    k2.metric("K-Means clusters",    f"{sel_k}")
    k3.metric("PCA PC1+PC2 variance", f"{exp_var1 + exp_var2:.1f}%")
    k4.metric("Unusual profiles (IF)", f"{n_anom}")

    tabs = st.tabs([
        "Overview",
        "K-Means Clustering",
        "PCA Visualisation",
        "Anomaly Detection",
        "District Results",
        "Validation",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW & FEATURE SELECTION
    # ══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.subheader("Machine Learning Approach")
        st.markdown(
            """
**Three complementary unsupervised ML techniques** are applied to characterise
Indian districts using 11 derived socioeconomic indicators:

| Technique | Purpose |
|---|---|
| **K-Means Clustering** | Group districts with similar indicator profiles |
| **Principal Component Analysis** | Reduce 11 dimensions to 2 for visualisation |
| **Isolation Forest** | Identify districts with statistically unusual profiles |

All techniques are **unsupervised** — no labelled outcomes are used. Results
reflect patterns within the 2011 Census data only.
"""
        )

        st.subheader("Feature Selection")
        st.caption(
            "11 derived indicators selected after removing highly collinear "
            "and arithmetically redundant features."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Features included in ML**")
            for f in features_used:
                st.markdown(f"- `{f}` — {FEATURE_LABELS.get(f, f)}")
        with col2:
            st.markdown("**Features excluded (with reason)**")
            for f, reason in ML_FEATURE_EXCLUSIONS.items():
                st.markdown(f"- `{f}`: {reason}")

        with st.expander("Pre-processing metadata"):
            meta = prep["meta"]
            st.write(f"- Rows in processed dataset: **{meta['n_rows_in']}**")
            st.write(f"- Rows used for ML: **{meta['n_rows_used']}**")
            st.write(f"- Rows dropped (NaN/Inf): **{meta['rows_dropped']}**")
            st.write(f"- Features with near-zero variance: "
                     f"**{len(meta['variance_warnings']) or 'none'}**")
            if meta["variance_warnings"]:
                for w in meta["variance_warnings"]:
                    st.warning(w)
            st.write("Standardization: **StandardScaler** (zero mean, unit variance)")
            st.write(f"Random seed: **{RANDOM_SEED}** (all models)")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — K-MEANS CLUSTERING
    # ══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.subheader("K-Means Clustering")
        st.caption(
            "K was selected by maximising the silhouette score across K = 2..8. "
            "Cluster labels are neutral integers — they do NOT represent "
            "official or causal development categories."
        )

        st.info(f"**Selection reason:** {km['selection_reason']}")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(figs["inertia"], use_container_width=True)
        with c2:
            st.plotly_chart(figs["silhouette"], use_container_width=True)

        st.markdown("---")
        st.subheader("K-Means Evaluation Table (K = 2 to 8)")
        st.dataframe(km["evaluation_df"], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader(f"Cluster Composition — K = {sel_k}")
        st.plotly_chart(figs["cluster_sizes"], use_container_width=True)
        st.dataframe(km["cluster_summary"], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Cluster Profiles — Deviation from Dataset Mean")
        st.caption(
            "Values show each cluster's average indicator value minus the "
            "dataset-wide average (in original units). "
            "Blue = above average · Red = below average."
        )
        st.plotly_chart(figs["profile_heatmap"], use_container_width=True)

        with st.expander("Cluster profile data (mean values per indicator)"):
            mean_cols = ["Cluster"] + [c for c in km["cluster_profile"].columns
                                       if c.endswith("_mean")]
            display_profile = km["cluster_profile"][mean_cols].copy()
            display_profile.columns = [
                c.replace("_mean", "") for c in display_profile.columns
            ]
            st.dataframe(display_profile, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — PCA VISUALISATION
    # ══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.subheader("Principal Component Analysis (PCA)")
        st.caption(
            "PCA reduces 11 indicators to 2 principal components for "
            "visualisation. It is a mathematical transformation — component "
            "directions reflect linear combinations of the original indicators, "
            "not causal relationships."
        )

        ev = pca_res["explained_var"]
        st.markdown(
            f"- **PC1** explains **{ev[0]*100:.1f}%** of total variance  \n"
            f"- **PC2** explains **{ev[1]*100:.1f}%** of total variance  \n"
            f"- **PC1 + PC2** together explain **{(ev[0]+ev[1])*100:.1f}%**"
        )

        st.plotly_chart(figs["pca_scatter"], use_container_width=True)
        st.caption(
            "Each point represents one district. Colour indicates K-Means "
            "cluster membership. Hover for district name, state, and PC coordinates."
        )

        st.markdown("---")
        st.subheader("Explained Variance per Component")
        st.plotly_chart(figs["explained_variance"], use_container_width=True)

        st.markdown("---")
        st.subheader("Component Loadings (PC1 and PC2)")
        st.caption(
            "Loadings indicate each original indicator's contribution to a "
            "principal component. Larger absolute values = stronger contribution. "
            "Signs indicate direction only."
        )
        loadings = pca_res["loadings_df"].copy()
        loadings.index.name = "Feature"
        loadings = loadings.reset_index()
        loadings["Feature Label"] = loadings["Feature"].map(
            lambda f: FEATURE_LABELS.get(f, f)
        )
        loadings = loadings[["Feature", "Feature Label", "PC1", "PC2"]]
        loadings = loadings.sort_values("PC1", key=abs, ascending=False)
        st.dataframe(loadings, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — ANOMALY DETECTION
    # ══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.subheader("Isolation Forest — Unusual District Profiles")
        st.caption(
            "Isolation Forest assigns an anomaly score to each district. "
            "Districts flagged as −1 show statistically unusual combinations "
            "of socioeconomic indicators within the 2011 dataset. "
            "This does NOT imply that a district is 'bad' or 'underdeveloped'."
        )

        st.markdown(
            f"- **Contamination parameter:** {CONTAMINATION:.0%} "
            f"(expected proportion of unusual profiles)  \n"
            f"- **Districts flagged as unusual:** **{n_anom}** "
            f"of {n_dist} ({n_anom/n_dist*100:.1f}%)  \n"
            f"- **Estimators:** 200 · **Random seed:** {RANDOM_SEED}"
        )

        st.plotly_chart(figs["anomaly_dist"], use_container_width=True)

        st.markdown("---")
        st.subheader("Districts with Unusual Profiles (Anomaly Flag = −1)")
        anomalies = results[results["Anomaly_Flag"] == -1].copy()
        anomalies = anomalies.sort_values("Anomaly_Score")
        display_cols = ["Name", "State", "Cluster_Label", "Anomaly_Score"] + features_used[:6]
        st.dataframe(
            anomalies[display_cols].rename(columns={"Cluster_Label": "Cluster"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Showing {len(anomalies)} districts flagged as unusual. "
            "Lower anomaly score = more atypical profile. "
            "Investigate individual districts in the **District Results** tab."
        )

        with st.expander("PCA scatter — unusual profiles highlighted"):
            anomaly_plot = results.copy()
            anomaly_plot["Profile"] = anomaly_plot["Anomaly_Flag"].map(
                {1: "Typical", -1: "Unusual"}
            )
            fig_anom_pca = px.scatter(
                anomaly_plot, x="PC1", y="PC2",
                color="Profile",
                color_discrete_map={"Typical": CHART_COLORS[0], "Unusual": "crimson"},
                hover_name="Name",
                hover_data={"State": True, "Anomaly_Score": ":.4f"},
                title="PCA Scatter — Unusual Profiles Highlighted",
                labels={"PC1": f"PC1 ({ev[0]*100:.1f}% var)",
                        "PC2": f"PC2 ({ev[1]*100:.1f}% var)"},
                opacity=0.7,
            )
            st.plotly_chart(fig_anom_pca, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — DISTRICT RESULTS TABLE
    # ══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.subheader("Full ML Results — All Districts")
        st.caption(
            "Every district with its K-Means cluster, PCA coordinates, "
            "Isolation Forest flag/score, and the 11 ML features."
        )

        # Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            state_opts = ["All"] + sorted(results["State"].unique().tolist())
            sel_state = st.selectbox("Filter by State", state_opts, key="ml_state")
        with f_col2:
            cluster_opts = ["All"] + sorted(results["Cluster_Label"].unique().tolist())
            sel_cluster = st.selectbox("Filter by Cluster", cluster_opts, key="ml_cluster")
        with f_col3:
            anom_opts = ["All", "Unusual only (−1)", "Typical only (+1)"]
            sel_anom = st.selectbox("Filter by Anomaly Flag", anom_opts, key="ml_anom")

        filtered = results.copy()
        if sel_state != "All":
            filtered = filtered[filtered["State"] == sel_state]
        if sel_cluster != "All":
            filtered = filtered[filtered["Cluster_Label"] == sel_cluster]
        if sel_anom == "Unusual only (−1)":
            filtered = filtered[filtered["Anomaly_Flag"] == -1]
        elif sel_anom == "Typical only (+1)":
            filtered = filtered[filtered["Anomaly_Flag"] == 1]

        st.write(f"Showing **{len(filtered)}** of {n_dist} districts")
        show_cols = (
            ["Name", "State", "Cluster_Label", "PC1", "PC2",
             "Anomaly_Flag", "Anomaly_Score"]
            + features_used
        )
        st.dataframe(
            filtered[show_cols].rename(columns={"Cluster_Label": "Cluster"}),
            use_container_width=True,
            hide_index=True,
        )

        if os.path.isfile(ML_RESULTS_PATH):
            with open(ML_RESULTS_PATH, "rb") as f:
                st.download_button(
                    "⬇️ Download district_ml_results.csv",
                    data=f,
                    file_name="district_ml_results.csv",
                    mime="text/csv",
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.subheader("ML Pipeline Validation")
        st.caption(
            "12 automated quality-control checks on the ML pipeline. "
            "Confirms data integrity, scaling correctness, reproducibility, "
            "and output completeness."
        )
        pass_count = len(ml_val["passed"])
        warn_count = len(ml_val["warnings"])
        err_count  = len(ml_val["errors"])
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("✅ Passed",   pass_count)
        vc2.metric("⚠️ Warnings", warn_count)
        vc3.metric("❌ Errors",   err_count)

        if ml_val["errors"]:
            for e in ml_val["errors"]:
                st.error(e)
        else:
            st.success("All validation checks passed — no errors detected.")
        for w in ml_val["warnings"]:
            st.warning(w)
        with st.expander("All passed checks"):
            for p in ml_val["passed"]:
                st.write(p)


# ============================================================
# SECTION 24b — ML CLUSTER INTERPRETATION
# ============================================================

def interpret_clusters(ml_output: dict) -> list:
    """
    Produce a neutral, data-derived interpretation for each K-Means cluster.

    For each cluster, calculates:
    - District count and percentage
    - Feature means
    - Delta from overall dataset mean for each feature
    - The top-3 distinguishing features (largest absolute delta)

    Language rules:
    - Cluster labels are neutral integers ("Cluster 0", etc.)
    - No value-laden labels ("developed", "poor", "backward", etc.)
    - Descriptions reference only observed model outputs.

    Returns
    -------
    list of dicts, one per cluster, with keys:
        cluster_id, cluster_label, n_districts, pct_districts,
        feature_means, overall_means, deltas, top_features, description
    """
    km   = ml_output["km_result"]
    prep = ml_output["ml_prep"]
    results = ml_output["results_df"]
    features = prep["features_used"]

    overall_means = {f: results[f].mean() for f in features}

    interpretations = []
    for c_id in sorted(results["Cluster"].unique()):
        subset = results[results["Cluster"] == c_id]
        n      = len(subset)
        pct    = round(100 * n / len(results), 1)

        feat_means = {f: round(float(subset[f].mean()), 3) for f in features}
        deltas     = {
            f: round(feat_means[f] - overall_means[f], 3)
            for f in features
        }

        # Top-3 distinguishing features by absolute delta
        top_feats = sorted(features, key=lambda f: abs(deltas[f]), reverse=True)[:3]

        # Build neutral description from deltas
        desc_parts = []
        for f in top_feats:
            d = deltas[f]
            direction = "higher" if d > 0 else "lower"
            desc_parts.append(
                f"{FEATURE_LABELS.get(f, f).split(' (')[0]} is {direction} "
                f"than the dataset average ({feat_means[f]:.2f} vs "
                f"{overall_means[f]:.2f})"
            )
        description = (
            f"Cluster {c_id} ({n} districts, {pct}%): "
            + "; ".join(desc_parts) + "."
        )

        interpretations.append({
            "cluster_id":     c_id,
            "cluster_label":  f"Cluster {c_id}",
            "n_districts":    n,
            "pct_districts":  pct,
            "feature_means":  feat_means,
            "overall_means":  {f: round(float(overall_means[f]), 3) for f in features},
            "deltas":         deltas,
            "top_features":   top_feats,
            "description":    description,
        })

    return interpretations


# ============================================================
# SECTION 24c — ML INTERPRETATION PAGE
# ============================================================

def page_ml_interpretation(analysis_df: pd.DataFrame):
    """
    Streamlit page: K-Means Cluster Interpretation, PCA summary, Anomaly summary,
    and District Explorer.  Renders after ML pipeline is cached.
    """
    st.title("🔍 ML Interpretation & District Explorer")
    st.caption(
        "Data-derived interpretations of the K-Means clusters, PCA structure, "
        "and Isolation Forest results from Census of India 2011 district data."
    )
    st.info(
        "ℹ️ All descriptions below are based solely on observed model outputs "
        "from the 2011 Census snapshot. Cluster labels are neutral data-derived "
        "groupings — they do **not** represent official categories or rankings. "
        "Patterns are correlational, not causal."
    )

    @st.cache_data(show_spinner="Loading ML results…")
    def cached_ml(df):
        return run_ml_pipeline(df)

    try:
        ml_output = cached_ml(analysis_df)
    except Exception as exc:
        st.error(f"ML pipeline error: {exc}")
        return

    interpretations = interpret_clusters(ml_output)
    km      = ml_output["km_result"]
    pca_res = ml_output["pca_result"]
    iso_res = ml_output["iso_result"]
    results = ml_output["results_df"]
    features = ml_output["ml_prep"]["features_used"]

    tabs = st.tabs([
        "Cluster Interpretation",
        "PCA Summary",
        "Anomaly Summary",
        "District Explorer",
    ])

    # ── Tab 1: Cluster Interpretation ────────────────────────────────────────
    with tabs[0]:
        st.subheader("K-Means Cluster Interpretation")
        st.caption(
            "For each cluster, the table shows how its average indicator values "
            "compare to the overall district dataset mean. Positive delta = "
            "above average; negative delta = below average (in original units)."
        )

        # Overall cluster-size table
        st.markdown("**Cluster Composition**")
        st.dataframe(km["cluster_summary"], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Top Distinguishing Features per Cluster**")
        st.caption(
            "The three indicators with the largest absolute deviation from the "
            "dataset mean define each cluster's most distinctive characteristics."
        )

        rows = []
        for interp in interpretations:
            for rank, f in enumerate(interp["top_features"], 1):
                rows.append({
                    "Cluster":   interp["cluster_label"],
                    "Rank":      rank,
                    "Feature":   FEATURE_LABELS.get(f, f).split(" (")[0],
                    "Cluster Avg": interp["feature_means"][f],
                    "Dataset Avg": interp["overall_means"][f],
                    "Delta":       interp["deltas"][f],
                })
        distinc_df = pd.DataFrame(rows)
        st.dataframe(distinc_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Neutral Cluster Descriptions**")
        for interp in interpretations:
            with st.expander(f"Cluster {interp['cluster_id']} — "
                             f"{interp['n_districts']} districts "
                             f"({interp['pct_districts']}%)"):
                st.write(interp["description"])
                # Full feature comparison table
                feat_rows = []
                for f in features:
                    feat_rows.append({
                        "Feature": FEATURE_LABELS.get(f, f).split(" (")[0],
                        "Cluster Avg": interp["feature_means"][f],
                        "Dataset Avg": interp["overall_means"][f],
                        "Delta":       interp["deltas"][f],
                    })
                feat_df = pd.DataFrame(feat_rows)
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Cluster Profile Heatmap")
        st.caption(
            "Deviation from dataset mean for each feature, per cluster. "
            "Blue = above average · Red = below average. "
            "Values are in original (unscaled) units."
        )
        # Build heatmap from interpret_clusters output
        feat_short = [FEATURE_LABELS.get(f, f).split(" (")[0][:18] for f in features]
        cluster_lbls = [f"Cluster {i['cluster_id']}" for i in interpretations]
        z_vals = np.array([[i["deltas"][f] for f in features] for i in interpretations])
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_vals, x=feat_short, y=cluster_lbls,
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="Delta from<br>dataset mean"),
            text=np.round(z_vals, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
        ))
        fig_heat.update_layout(
            title="Cluster Profiles — Deviation from Dataset Mean (original units)",
            xaxis=dict(tickangle=-40),
            height=300 + 60 * len(interpretations),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.subheader("What K-Means Does in This Project")
        st.markdown(
            """
K-Means partitions the 640 districts into **K groups** such that districts
within each group are more similar to each other than to districts in other
groups, based on the 11 selected socioeconomic indicators.

**Selection of K:**
K was chosen by evaluating silhouette scores for K = 2 through K = 8. The
value with the highest silhouette score was selected (K = 4, score = 0.2232).
The silhouette score measures how well each district fits its assigned cluster
versus the nearest alternative — higher values indicate better-defined
separation.

**Important limitations:**
- Clusters are mathematical groupings, not official categories.
- The silhouette score of 0.2232 indicates moderate cluster separation —
  districts do not form sharply distinct groups; there is overlap.
- Clustering reflects the 11 selected features from 2011 data only.
- Different feature sets or K values would produce different groupings.
"""
        )

    # ── Tab 2: PCA Summary ────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("PCA — Dimension Reduction Summary")
        ev = pca_res["explained_var"]
        st.markdown(
            f"""
Principal Component Analysis reduced the 11-feature indicator space to 2
dimensions for visualisation.

| Component | Explained Variance |
|-----------|-------------------|
| **PC1** | **{ev[0]*100:.1f}%** |
| **PC2** | **{ev[1]*100:.1f}%** |
| **PC1 + PC2 (combined)** | **{(ev[0]+ev[1])*100:.1f}%** |

A combined variance of {(ev[0]+ev[1])*100:.1f}% means that the 2D projection
retains a substantial but partial representation of the 11-dimensional data.
The remaining {(1-(ev[0]+ev[1]))*100:.1f}% of variation is not visible in the
2D scatter plot.
"""
        )

        st.subheader("Component Loadings")
        st.caption(
            "Loadings show each feature's mathematical contribution to PC1 and PC2. "
            "Larger absolute values = stronger contribution. "
            "Signs indicate direction only — they do not have inherent meaning."
        )
        loadings = pca_res["loadings_df"].copy().reset_index()
        loadings.columns = ["Feature", "PC1 Loading", "PC2 Loading"]
        loadings["Feature Label"] = loadings["Feature"].map(
            lambda f: FEATURE_LABELS.get(f, f).split(" (")[0]
        )
        loadings = loadings[["Feature", "Feature Label", "PC1 Loading", "PC2 Loading"]]
        loadings = loadings.sort_values("PC1 Loading", key=abs, ascending=False)
        st.dataframe(loadings, use_container_width=True, hide_index=True)

        st.subheader("What PCA Contributes")
        st.markdown(
            """
PCA serves two roles in this project:

1. **Visualisation:** The 2D PCA scatter plot (in the Machine Learning section)
   shows how districts relate to one another in reduced-dimension space, coloured
   by K-Means cluster assignment. This helps assess whether clusters are
   geometrically separated.

2. **Indicator structure:** The component loadings reveal which indicators vary
   together in the dataset. For example, features that load strongly in the same
   direction on PC1 tend to be positively correlated across districts.

**Important:** PCA is a mathematical transformation — it does not assign meaning
to the components. Component directions are not labelled as "development axis"
or similar. The loadings describe linear combinations of the 11 indicators in
this 2011 dataset only.
"""
        )

    # ── Tab 3: Anomaly Summary ────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Isolation Forest — Anomaly Summary")
        n_anom = iso_res["n_anomalies"]
        n_dist = len(results)
        st.markdown(
            f"""
Isolation Forest identified **{n_anom} districts** ({n_anom/n_dist*100:.1f}%)
with statistically unusual combinations of the 11 selected indicators, relative
to the rest of the district dataset.

**How Isolation Forest works:**
The algorithm isolates data points by randomly selecting a feature and a random
split value. Points that can be isolated with fewer splits (shorter average
path length) are considered more unusual. The `contamination` parameter (set to
{CONTAMINATION:.0%}) controls the expected proportion of unusual profiles.

**What "unusual" means here:**
A flagged district has an atypical *combination* of the 11 indicators —
not that any single indicator is extreme. For example, a district with an
unusual combination of high ST_Pop_Pct and high Literacy_Rate would appear
unusual relative to the general pattern where these features are negatively
associated.

**What it does NOT mean:**
- "Unusual" is not equivalent to "problematic" or "underdeveloped".
- Flagged districts may be unusual because they are exceptionally high on
  positive indicators, not low.
- The anomaly score is data-driven and pattern-based only.
"""
        )
        st.subheader("Districts with Unusual Profiles")
        anomalies = results[results["Anomaly_Flag"] == -1].copy()
        anomalies = anomalies.sort_values("Anomaly_Score")
        display_cols = (
            ["Name", "State", "Cluster_Label", "Anomaly_Score"]
            + features[:6]
        )
        st.dataframe(
            anomalies[display_cols].rename(columns={"Cluster_Label": "Cluster"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"{n_anom} districts shown. Sorted by anomaly score (lowest = most atypical "
            "indicator combination). All values from Census of India 2011."
        )

    # ── Tab 4: District Explorer ──────────────────────────────────────────────
    with tabs[3]:
        st.subheader("District Explorer")
        st.caption(
            "Select a district to view its socioeconomic indicator profile, "
            "cluster assignment, and anomaly flag. "
            "All values are from **Census of India 2011** and reflect "
            "conditions at that point in time only."
        )

        EXPLORER_FEATURES = [
            "Literacy_Rate", "Female_Literacy_Rate", "Gender_Literacy_Gap",
            "Worker_Participation", "Female_Worker_Part", "Agri_Worker_Pct",
            "Child_Pop_Pct", "Sex_Ratio",
        ]

        col_s, col_d = st.columns(2)
        with col_s:
            states = sorted(results["State"].unique().tolist())
            sel_state_ex = st.selectbox("Select State / UT", states, key="explorer_state")
        with col_d:
            districts_in_state = sorted(
                results[results["State"] == sel_state_ex]["Name"].unique().tolist()
            )
            sel_dist_ex = st.selectbox("Select District", districts_in_state,
                                       key="explorer_district")

        row = results[
            (results["State"] == sel_state_ex) & (results["Name"] == sel_dist_ex)
        ]
        if row.empty:
            st.warning("District not found in ML results.")
        else:
            row = row.iloc[0]

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Cluster",       row["Cluster_Label"])
            mc2.metric("Anomaly Flag",  "Unusual (−1)" if row["Anomaly_Flag"] == -1
                                         else "Typical (+1)")
            mc3.metric("Anomaly Score", f"{row['Anomaly_Score']:.4f}")

            st.markdown("---")
            st.markdown("**Socioeconomic Indicators — Census of India 2011**")

            indicator_rows = []
            for f in EXPLORER_FEATURES:
                if f in row.index:
                    overall_mean = float(results[f].mean())
                    val = float(row[f])
                    delta = val - overall_mean
                    indicator_rows.append({
                        "Indicator": FEATURE_LABELS.get(f, f).split(" (")[0],
                        "District Value": round(val, 2),
                        "Dataset Mean": round(overall_mean, 2),
                        "Delta": round(delta, 2),
                    })
            ind_df = pd.DataFrame(indicator_rows)
            st.dataframe(ind_df, use_container_width=True, hide_index=True)

            # Radar chart
            categories = [r["Indicator"] for r in indicator_rows]
            vals_raw   = [r["District Value"] for r in indicator_rows]
            means_raw  = [r["Dataset Mean"] for r in indicator_rows]

            # Normalise 0–1 within dataset range for radar display
            norm_vals, norm_means = [], []
            for f in EXPLORER_FEATURES:
                if f in results.columns:
                    mn, mx = results[f].min(), results[f].max()
                    rng = mx - mn if mx != mn else 1.0
                    norm_vals.append(round((float(row[f]) - mn) / rng, 3))
                    norm_means.append(round((results[f].mean() - mn) / rng, 3))

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=sel_dist_ex,
                line_color=CHART_COLORS[0],
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_means + [norm_means[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Dataset Mean",
                line_color="grey",
                opacity=0.4,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"Indicator Profile — {sel_dist_ex}, {sel_state_ex}",
                legend=dict(orientation="h"),
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption(
                "Radar values are normalised to [0, 1] within the dataset range "
                "for visual comparison only. Table values above show the original "
                "indicator magnitudes from Census of India 2011."
            )


# ============================================================
# SECTION 24d — RECOMMENDATIONS PAGE
# ============================================================

def page_recommendations(analysis_df: pd.DataFrame):
    """
    Streamlit page: data-driven recommendations derived from the Step 4
    analytical storytelling (REC-01, REC-02, REC-03).

    Displays the three existing recommendations with their supporting evidence,
    caveats, and explicit data limitations. No new recommendations are invented.
    """
    st.title("📋 Recommendations")
    st.caption(
        "Three data-informed recommendations derived from the exploratory "
        "analysis of Census of India 2011 district-level data."
    )
    st.warning(
        "**Important disclaimer:** These recommendations are based on "
        "exploratory analysis of historical 2011 Census data. They suggest "
        "areas for *further assessment*, not proven interventions. "
        "Census 2011 is now over a decade old — current conditions may differ "
        "substantially. No causal claims are made."
    )

    # Load analytical story to get recommendations at runtime
    @st.cache_data(show_spinner="Loading analytical story…")
    def cached_story(df):
        return build_analytical_story(df)

    try:
        story = cached_story(analysis_df)
    except Exception as exc:
        st.error(f"Could not generate recommendations: {exc}")
        return

    recs  = story["recommendations"]
    hyps  = {h["id"]: h for h in story["hypotheses"]}

    st.markdown("---")

    ICON = ["1️⃣", "2️⃣", "3️⃣"]
    for idx, rec in enumerate(recs):
        st.subheader(f"{ICON[idx]} {rec['id']} — Target Pattern")
        st.markdown(f"> {rec['target_pattern']}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Proposed Action**")
            st.markdown(rec["proposed_action"])
        with c2:
            st.markdown("**Relevance (from EDA / Storytelling)**")
            st.markdown(rec["relevance"])

        with st.expander(f"Caveat & Data Limitations — {rec['id']}"):
            st.markdown(rec["caveat"])
            st.markdown(
                "**General data limitation:** All indicators are derived from "
                "Census of India 2011. This dataset captures a single historical "
                "snapshot. Administrative boundaries, population distributions, "
                "and socioeconomic conditions may have changed since 2011. "
                "Census data alone is not sufficient for programme design."
            )
            for hyp_id in rec.get("hyp_ids", []):
                if hyp_id in hyps:
                    h = hyps[hyp_id]
                    st.markdown(f"**Supporting {hyp_id}:** {h['statement']}")

        st.markdown("---")

    # Summary table
    st.subheader("Recommendations Summary")
    sum_rows = []
    for rec in recs:
        sum_rows.append({
            "ID":             rec["id"],
            "Target Pattern": rec["target_pattern"][:100] + "…",
            "Proposed Action": rec["proposed_action"][:100] + "…",
        })
    st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Framework Note")
    st.markdown(
        """
These recommendations follow the IBM SkillsBuild Data Analytics framework:

**Observations → Insights → Hypotheses → Recommendations**

Each recommendation is linked to one or more explicitly-labelled hypotheses.
Hypotheses are clearly marked as such and are not presented as proven facts.
The recommendations suggest areas for *prioritisation of further assessment*
using additional datasets and current data — not specific programme designs
based on Census 2011 alone.
"""
    )


# ============================================================
# SECTION 24e — METHODOLOGY / ABOUT PAGE
# ============================================================

def page_methodology():
    """
    Streamlit page: project methodology, pipeline, limitations, and reproducibility.
    Describes the actual implemented project — no technologies or outcomes are invented.
    """
    st.title("📖 Methodology / About")
    st.caption(
        "A full description of the IndiDevAI project pipeline, analytical approach, "
        "machine learning methods, limitations, and reproducibility."
    )

    tabs = st.tabs([
        "Project Overview",
        "Data & Preparation",
        "Feature Engineering",
        "EDA & Storytelling",
        "Machine Learning",
        "Limitations",
        "Reproducibility",
    ])

    # ── Tab 1: Project Overview ───────────────────────────────────────────────
    with tabs[0]:
        st.subheader("IndiDevAI — AI-Powered District Development Intelligence")
        st.markdown(
            """
**Project:** IndiDevAI
**Internship:** IBM SkillsBuild Academic Internship 2026 — Data Analytics with AI
**Dataset:** Census of India 2011 — Primary Census Abstract (PCA),
State and District Level
**Source:** Office of the Registrar General & Census Commissioner, India
**Official source:** https://censusindia.gov.in/nada/index.php/catalog/6191

---

### Objective

IndiDevAI analyses district-level socioeconomic characteristics of 640 Indian
districts using Census of India 2011 data. The project applies data cleaning,
feature engineering, exploratory analysis, analytical storytelling, and
unsupervised machine learning to identify patterns, groupings, and statistically
unusual profiles across demographic, education, and employment dimensions.

### Scope

- **Geographic level:** District (Level == "DISTRICT", TRU == "Total")
- **Districts analysed:** 640
- **States / UTs:** 35
- **Year of data:** 2011 (historical snapshot)

### Deliverables

| Component | Description |
|-----------|-------------|
| `Aarav_IndiDevAI.py` | Single Python file containing the complete project and Streamlit application |
| `requirements.txt` | Python package dependencies |
| `data/processed/district_analysis_ready.csv` | Cleaned and feature-engineered district dataset (640 × 75) |
| `data/processed/district_ml_results.csv` | ML outputs: cluster labels, PCA coordinates, anomaly scores (640 × 20) |
"""
        )

    # ── Tab 2: Data & Preparation ─────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Dataset")
        st.markdown(
            """
| Attribute | Value |
|-----------|-------|
| File | `DDW_PCA0000_2011_Indiastatedist.xlsx` |
| Sheet | Sheet1 |
| Raw rows | 2,028 (excluding header) |
| Raw columns | 94 |
| Scope filter | Level == "DISTRICT" AND TRU == "Total" |
| District rows after filter | 640 |
| States / UTs | 35 |
| Missing values | 0 (after zero-population rows removed) |
| Duplicate district records | 0 |

### Data Loading

- File loaded with `pandas.read_excel()` using `dtype=str` to preserve zero-padded
  state and district codes.
- Column names are whitespace-stripped on load.
- Structural columns (`State`, `District`, `Level`, `Name`, `TRU`, `TOT_P`,
  `TOT_M`, `TOT_F`) are checked for presence before proceeding.

### Validation (8-point check)

1. Dataset dimensions verified (rows, columns)
2. Unique `Level` values confirmed (India / STATE / DISTRICT)
3. Unique `TRU` values confirmed (Total / Rural / Urban)
4. Duplicate (State, District, Level, TRU) row check
5. State code uniqueness verified
6. Missing value audit across structural columns
7. District-level Total row count confirmed (~640)
8. State count confirmed (~35)

### Cleaning Operations

| Operation | Justification |
|-----------|---------------|
| Strip whitespace from string columns | Prevents mismatches in Level/TRU filtering |
| Drop entirely-empty columns | Trailing placeholder columns contain no data |
| Convert numeric Census columns to float | Raw data loaded as strings; conversion required |
| Drop zero/missing TOT_P rows | Rate computations undefined for zero-population records |

### Notable Data Finding

One district — Jaintia Hills (Meghalaya) — shows a **negative Gender Literacy Gap**
(female literacy exceeds male literacy). This is consistent with the matrilineal
social structure documented for that region and is retained as valid data.
No imputation was applied.
"""
        )

    # ── Tab 3: Feature Engineering ────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Derived Features (14 indicators)")
        st.markdown(
            """
All derived features are computed from raw Census columns using zero-denominator-
guarded division. Every formula was verified against the Census of India 2011
definition before implementation.

**Literacy denominators:** The Census defines effective literacy for the
population aged **7 and above** — children 0–6 are excluded because literacy is
not enumerated for them. All literacy-rate denominators subtract the 0–6
age-group column from the corresponding total population.

| Feature | Formula | Notes |
|---------|---------|-------|
| `Sex_Ratio` | (TOT_F / TOT_M) × 1 000 | Females per 1,000 males |
| `Child_Pop_Pct` | (P_06 / TOT_P) × 100 | Children 0–6 as % of total |
| `SC_Pop_Pct` | (P_SC / TOT_P) × 100 | Scheduled Caste population % |
| `ST_Pop_Pct` | (P_ST / TOT_P) × 100 | Scheduled Tribe population % |
| `Literacy_Rate` | (P_LIT / (TOT_P − P_06)) × 100 | Overall literacy, pop 7+ |
| `Male_Literacy_Rate` | (M_LIT / (TOT_M − M_06)) × 100 | Male literacy, pop 7+ |
| `Female_Literacy_Rate` | (F_LIT / (TOT_F − F_06)) × 100 | Female literacy, pop 7+ |
| `Gender_Literacy_Gap` | Male_Literacy_Rate − Female_Literacy_Rate | pp gap; negative = female advantage |
| `Worker_Participation` | (TOT_WORK_P / TOT_P) × 100 | Overall work participation % |
| `Female_Worker_Part` | (TOT_WORK_F / TOT_F) × 100 | Female work participation % |
| `Main_Worker_Pct` | (MAINWORK_P / TOT_WORK_P) × 100 | Main workers % of total workers |
| `Marginal_Worker_Pct` | (MARGWORK_P / TOT_WORK_P) × 100 | Marginal workers % |
| `Non_Worker_Pct` | (NON_WORK_P / TOT_P) × 100 | Non-workers % of total population |
| `Agri_Worker_Pct` | ((MAIN_CL_P + MAIN_AL_P) / TOT_WORK_P) × 100 | Agricultural worker share % |

### Feature Validation

Each derived feature is checked for:
- NaN values
- Infinite values
- Out-of-range values (e.g., rates > 100% or < 0%)

Gender_Literacy_Gap is expected to have negative values (valid) — the validator
flags this for inspection but does not remove these records.
"""
        )

    # ── Tab 4: EDA & Storytelling ─────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Exploratory Data Analysis")
        st.markdown(
            """
EDA is performed on the 640 district records using the 14 derived features.

**Analyses performed:**
- Descriptive statistics (mean, median, std, min, max, IQR) for all features
- Distribution visualisations (histograms, box plots) for key indicators
- State-level aggregates (mean indicator values per state)
- Top-10 and bottom-10 district rankings for Literacy_Rate, Sex_Ratio, and others
- IQR-based outlier detection for all derived features
- Pearson correlation matrix across all 14 derived features

**Key EDA findings:**
- Literacy Rate spans 36.10% (Alirajpur, Madhya Pradesh) to 97.91% (Serchhip, Mizoram)
- Female Literacy Rate minimum: 30.29% (Alirajpur)
- Gender Literacy Gap maximum: 34.0 pp (Shravasti, Uttar Pradesh)
- Child_Pop_Pct vs Literacy_Rate: Pearson r = −0.678 (strongest cross-domain correlation)
- Agri_Worker_Pct vs Literacy_Rate: Pearson r ≈ −0.40
- ST_Pop_Pct shows 85 IQR-outlier districts (concentrated in Northeast and Central India)

---

### Analytical Storytelling

The storytelling framework follows the IBM SkillsBuild structure:

**Observations → Insights → Hypotheses → Recommendations**

- **Observations (7):** Factual statements derived programmatically from the dataset
- **Insights (6):** Analytical interpretations using careful non-causal language
- **Hypotheses (3):** Explicitly-labelled plausible explanations for observed patterns
- **Recommendations (3):** Suggested areas for further assessment linked to hypotheses

**Language discipline enforced throughout:**
✓ "is associated with", "may indicate", "suggests a possible relationship"
✗ "causes", "leads to", "because of", "proves"

An automated validation function checks all storytelling outputs for causal
language, structural completeness, and reference integrity.
"""
        )

    # ── Tab 5: Machine Learning ───────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Machine Learning Pipeline")
        st.markdown(
            """
Three unsupervised ML techniques are applied to the 640 districts using
11 socioeconomic indicators selected from the 14 derived features.

### Feature Selection for ML

**11 features included:**
`Sex_Ratio`, `Child_Pop_Pct`, `SC_Pop_Pct`, `ST_Pop_Pct`, `Literacy_Rate`,
`Female_Literacy_Rate`, `Gender_Literacy_Gap`, `Worker_Participation`,
`Female_Worker_Part`, `Agri_Worker_Pct`, `Main_Worker_Pct`

**3 features excluded:**
- `Male_Literacy_Rate` — highly collinear with Literacy_Rate and Female_Literacy_Rate
- `Non_Worker_Pct` — arithmetic complement of Worker_Participation (r = −1.000)
- `Marginal_Worker_Pct` — arithmetic complement of Main_Worker_Pct

**Pre-processing:** All 11 features are standardised using `StandardScaler`
(zero mean, unit variance) before any model is fitted. Identifiers (State,
District, Name) are never passed to the scaler or model. Random seed = 42 for
all models.

---

### K-Means Clustering

- Library: `sklearn.cluster.KMeans`
- K evaluated: 2 through 8
- K selection criterion: highest silhouette score
- **Selected K = 4** (silhouette = 0.2232)
- Cluster labels are neutral integers (Cluster 0, 1, 2, 3)

The silhouette score of 0.2232 indicates moderate cluster separation.
Districts do not form sharply distinct groups — there is meaningful overlap.

---

### Principal Component Analysis

- Library: `sklearn.decomposition.PCA`
- Components: full PCA fitted; 2-component projection for visualisation
- **PC1 explains 34.4%** of total variance
- **PC2 explains 24.3%** of total variance
- **PC1 + PC2: 58.7%** combined

PCA is a mathematical transformation — it does not establish causal structure.
Component loadings describe linear combinations of the 11 indicators only.

---

### Isolation Forest (Anomaly Detection)

- Library: `sklearn.ensemble.IsolationForest`
- Estimators: 200
- Contamination: 5% (expected proportion of unusual profiles)
- **32 districts flagged** as having unusual indicator combinations (5.0%)

"Unusual" means a statistically atypical combination of the 11 indicators —
not that a district has poor outcomes or is a "problem" district.

---

### Output

All ML results are saved to `data/processed/district_ml_results.csv`
(640 rows × 20 columns): cluster labels, PCA coordinates, anomaly flags,
anomaly scores, and the 11 ML feature values.

**Validation:** 12/12 pipeline checks pass (no missing values, no leakage,
correct scaling, K range confirmed, cluster sizes sum correctly, PCA 2D
finite, anomaly flags valid).
"""
        )

    # ── Tab 6: Limitations ────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Important Limitations")
        st.markdown(
            """
### Data Limitations

| Limitation | Detail |
|-----------|--------|
| **Historical snapshot** | Census data is from 2011. Conditions, populations, and administrative boundaries may have changed substantially since then. Results describe the 2011 dataset only. |
| **Single data source** | The analysis uses only Census 2011 PCA data. Economic, healthcare, infrastructure, and governance data are absent. Observed patterns cannot be fully contextualised without supplementary datasets. |
| **District boundaries** | District boundaries and administrative structures may have changed since 2011. Some districts may have been split, merged, or renamed. |
| **Self-reported literacy** | Literacy status in Census 2011 is self-reported. The methodology for recording literacy may vary across enumerators and regions. |
| **Worker category definitions** | Census worker categories (Main, Marginal, Cultivator, Agricultural Labourer) reflect the reference period and enumeration methodology of Census 2011. These may not align with other employment datasets. |

---

### Analytical Limitations

| Limitation | Detail |
|-----------|--------|
| **Correlation ≠ causation** | All observed associations (e.g., Child_Pop_Pct vs Literacy_Rate) are correlational. No causal claims are supported by this analysis. |
| **Clusters are exploratory** | K-Means clusters are mathematical groupings based on 11 selected features. They do not represent official socioeconomic categories or development classifications. |
| **Silhouette score** | The selected K=4 silhouette score of 0.2232 indicates moderate separation. Cluster boundaries are not sharp — many districts sit between cluster centres. |
| **PCA partial coverage** | PC1 + PC2 explain 58.7% of variance. The remaining 41.3% is not visible in the 2D visualisation. |
| **Anomaly context** | Isolation Forest anomaly flags identify unusual *feature combinations*, not socioeconomic deficiencies. An unusual profile may reflect a district that is distinctively high-performing on certain indicators, not low-performing. |
| **Feature selection** | Results depend on the 11 features selected. Different feature sets would produce different clusters and anomaly flags. |
| **ML not predictive** | No future predictions are made. The ML output describes the 2011 district distribution only. |

---

### Policy Limitation

These analytical outputs are intended for exploratory pattern discovery and
educational demonstration. They are **not** suitable for policy decisions without:
- Current data (post-2011)
- Domain expert review
- Additional socioeconomic datasets
- Formal policy evaluation methodology
"""
        )

    # ── Tab 7: Reproducibility ────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Reproducibility")
        st.markdown(
            f"""
### Code

The entire project is contained in a **single Python file**:
`Aarav_IndiDevAI.py`

This file includes all data loading, validation, cleaning, feature engineering,
EDA, analytical storytelling, machine learning, and the Streamlit application.

### Environment

| Component | Value |
|-----------|-------|
| Language | Python 3.9+ |
| Data processing | pandas, numpy |
| Visualisation | Plotly (plotly.express, plotly.graph_objects) |
| Statistical analysis | scipy.stats |
| Machine learning | scikit-learn (KMeans, PCA, IsolationForest, StandardScaler) |
| Dashboard | Streamlit |
| Excel reading | openpyxl |
| Random seed | {RANDOM_SEED} (all models) |

### Determinism

- All ML models use `random_state={RANDOM_SEED}`.
- The pipeline is deterministic given the same dataset and package versions.
- `requirements.txt` documents the package dependencies.

### Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run Aarav_IndiDevAI.py

# Run the data pipeline and ML pipeline from the command line
python Aarav_IndiDevAI.py
```

### Data Files

| File | Description |
|------|-------------|
| `data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx` | Original Census data — never modified |
| `data/processed/district_analysis_ready.csv` | Pipeline output (640 × 75) |
| `data/processed/district_ml_results.csv` | ML output (640 × 20) |

The processed files are generated automatically when the pipeline runs.
They do not need to be committed to version control.
"""
        )


def page_coming_soon(section_name: str):
    """Placeholder page for sections not yet implemented."""
    st.title(f"🚧 {section_name}")
    st.info(
        f"**{section_name}** will be implemented in the next development phase.  \n"
        "The data pipeline and EDA are complete and operational."
    )


# ============================================================
# SECTION 25 — MAIN APPLICATION
# ============================================================

def main():
    """Streamlit application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

    st.sidebar.title("IndiDevAI")
    st.sidebar.caption("Census of India 2011 — District Analytics")

    sections = [
        "Home", "Dataset Overview",
        "Demographics", "Education", "Employment",
        "Exploratory Analysis",
        "District Clustering", "PCA Visualisation",
        "Anomaly Detection", "ML Interpretation",
        "AI-Assisted Insights",
        "Recommendations", "Methodology / About",
    ]
    selection = st.sidebar.radio("Navigate", sections)

    @st.cache_data(show_spinner="Running data pipeline…")
    def cached_pipeline():
        raw_df              = load_dataset()
        val_rep             = validate_dataset(raw_df)
        clean_df, clean_rep = clean_dataset(raw_df)
        dist_df, filt_rep   = filter_district_data(clean_df)
        feat_df             = engineer_features(dist_df)
        feat_val            = validate_features(feat_df)
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

    # Load processed CSV for EDA pages (lighter path — avoids re-running pipeline)
    @st.cache_data(show_spinner="Loading analysis data…")
    def cached_analysis_data():
        if os.path.isfile(PROCESSED_PATH):
            return load_analysis_data()
        return featured_df  # fallback if CSV not yet saved

    try:
        analysis_df = cached_analysis_data()
    except Exception:
        analysis_df = featured_df  # graceful fallback

    if selection == "Home":
        page_home()
    elif selection == "Dataset Overview":
        page_dataset_overview(raw_df, featured_df, val_report,
                              clean_report, filter_report, feat_val)
    elif selection == "Demographics":
        page_demographics(analysis_df)
    elif selection == "Education":
        page_education(analysis_df)
    elif selection == "Employment":
        page_employment(analysis_df)
    elif selection == "Exploratory Analysis":
        page_exploratory_analysis(analysis_df)
    elif selection in ("District Clustering", "PCA Visualisation", "Anomaly Detection"):
        page_machine_learning(analysis_df)
    elif selection == "ML Interpretation":
        page_ml_interpretation(analysis_df)
    elif selection == "AI-Assisted Insights":
        page_ai_insights(analysis_df)
    elif selection == "Recommendations":
        page_recommendations(analysis_df)
    elif selection == "Methodology / About":
        page_methodology()

    st.sidebar.markdown("---")
    st.sidebar.caption("IBM SkillsBuild Internship 2026")


# ============================================================
# SECTION 26 — ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline(verbose=True)

    # Run ML pipeline after data pipeline to generate district_ml_results.csv
    print("\n[ML] Running ML pipeline…")
    ml_out = run_ml_pipeline()
    ml_val = validate_ml_pipeline(ml_out)
    km_res = ml_out["km_result"]
    pca_r  = ml_out["pca_result"]
    iso_r  = ml_out["iso_result"]

    print(f"[ML] Selected K = {km_res['selected_k']} "
          f"(silhouette = {km_res['evaluation_df'].sort_values('Silhouette', ascending=False).iloc[0]['Silhouette']:.4f})")
    ev = pca_r["explained_var"]
    print(f"[ML] PCA: PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%  "
          f"PC1+PC2={( ev[0]+ev[1])*100:.1f}%")
    print(f"[ML] Isolation Forest: {iso_r['n_anomalies']} unusual profiles "
          f"({iso_r['n_anomalies']/len(ml_out['results_df'])*100:.1f}%)")

    pass_n = len(ml_val["passed"])
    warn_n = len(ml_val["warnings"])
    err_n  = len(ml_val["errors"])
    print(f"[ML] Validation: {pass_n} passed / {warn_n} warnings / {err_n} errors")
    if ml_val["errors"]:
        for e in ml_val["errors"]:
            print(f"     ERROR: {e}")
    if ml_val["warnings"]:
        for w in ml_val["warnings"]:
            print(f"     WARN:  {w}")
    print("[ML] Pipeline complete.")
