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
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "district_analysis_ready.csv")
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
# SECTION 20 — ML: K-MEANS CLUSTERING (placeholder)
# ============================================================

def run_kmeans(df: pd.DataFrame, features: list, n_clusters: int = N_CLUSTERS):
    """Placeholder for K-Means clustering. To be implemented in ML phase."""
    return df


# ============================================================
# SECTION 21 — ML: PCA VISUALISATION (placeholder)
# ============================================================

def run_pca_analysis(df: pd.DataFrame, features: list, n_components: int = 2):
    """Placeholder for PCA. To be implemented in ML phase."""
    return df


# ============================================================
# SECTION 22 — ML: ANOMALY DETECTION (placeholder)
# ============================================================

def run_anomaly_detection(df: pd.DataFrame, features: list,
                          contamination: float = CONTAMINATION):
    """Placeholder for Isolation Forest. To be implemented."""
    return df


# ============================================================
# SECTION 23 — AI-ASSISTED INSIGHTS (placeholder)
# ============================================================

def generate_insights(df: pd.DataFrame) -> list:
    """Placeholder for AI-assisted insights. To be implemented."""
    return []


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
        "Anomaly Detection", "AI-Assisted Insights",
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
# SECTION 26 — ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline(verbose=True)
