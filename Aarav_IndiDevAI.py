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
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# SECTION 2 — CONFIGURATION
# ============================================================

# Reproducibility seed used for all random-state parameters
RANDOM_SEED = 42

# Path to the raw dataset (relative to project root)
DATA_PATH = os.path.join("data", "raw", "DDW_PCA0000_2011_Indiastatedist.xlsx")

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


# ============================================================
# SECTION 3 — DATA LOADING
# ============================================================

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Census PCA Excel file into a DataFrame.

    Returns the full dataset (all geographic levels, all TRU values)
    without any filtering, so the caller can inspect the raw structure.

    Parameters
    ----------
    path : str
        Relative path to the .xlsx file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all rows and columns from Sheet1.
    """
    df = pd.read_excel(path, sheet_name="Sheet1", dtype=str)
    return df


def get_district_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the raw DataFrame to district-level Total records.

    Applies:
        Level == 'DISTRICT'   (column 'Level')
        TRU   == 'Total'      (column 'TRU')

    Numeric columns are cast to numeric after filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by load_raw_data().

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only district-level Total rows,
        with numeric columns converted to float.
    """
    mask = (df["Level"].str.strip() == TARGET_LEVEL) & \
           (df["TRU"].str.strip()   == TARGET_TRU)
    district_df = df[mask].copy()

    # Convert numeric census columns (J onward) to float
    numeric_cols = district_df.columns[9:]          # columns after 'TRU'
    district_df[numeric_cols] = district_df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    district_df.reset_index(drop=True, inplace=True)
    return district_df


# ============================================================
# SECTION 4 — DATA VALIDATION
# ============================================================

def validate_data(df: pd.DataFrame) -> dict:
    """
    Run basic data quality checks on the raw DataFrame.

    Checks include:
    - Row and column counts
    - Unique values in 'Level' and 'TRU'
    - Missing-value counts per column
    - Duplicate row count

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (output of load_raw_data).

    Returns
    -------
    dict
        Dictionary with validation results.
    """
    report = {
        "total_rows":        len(df),
        "total_columns":     len(df.columns),
        "unique_levels":     df["Level"].unique().tolist() if "Level" in df.columns else [],
        "unique_tru":        df["TRU"].unique().tolist()   if "TRU"   in df.columns else [],
        "missing_per_col":   df.isnull().sum().to_dict(),
        "total_missing":     int(df.isnull().sum().sum()),
        "duplicate_rows":    int(df.duplicated().sum()),
        "district_total_rows": int(
            ((df.get("Level", pd.Series(dtype=str)).str.strip() == TARGET_LEVEL) &
             (df.get("TRU",   pd.Series(dtype=str)).str.strip() == TARGET_TRU)).sum()
        ),
    }
    return report


# ============================================================
# SECTION 5 — DATA CLEANING
# ============================================================

def clean_district_data(district_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the district-level DataFrame.

    Steps applied:
    1. Strip leading/trailing whitespace from string columns.
    2. Standardise 'Name' column as the district label.
    3. Drop rows where total population (TOT_P) is zero or missing,
       as they cannot meaningfully participate in analysis.
    4. Reset the index.

    Parameters
    ----------
    district_df : pd.DataFrame
        Filtered district DataFrame (output of get_district_data).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = district_df.copy()

    # Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Remove records with no population data
    if "TOT_P" in df.columns:
        df = df[df["TOT_P"].notna() & (df["TOT_P"] > 0)]

    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# SECTION 6 — FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive analytical indicators from raw census columns.

    Derived columns (computed only when source columns are present):
    - Sex_Ratio              : females per 1 000 males  (TOT_F / TOT_M * 1000)
    - Literacy_Rate          : literate population / (TOT_P - P_06) * 100
    - Female_Literacy_Rate   : F_LIT / (TOT_F - F_06) * 100
    - Gender_Literacy_Gap    : male literacy rate minus female literacy rate
    - Worker_Participation   : TOT_WORK_P / TOT_P * 100
    - Female_Worker_Part     : TOT_WORK_F / TOT_F * 100
    - Child_Pop_Pct          : P_06 / TOT_P * 100
    - SC_Pop_Pct             : P_SC / TOT_P * 100
    - ST_Pop_Pct             : P_ST / TOT_P * 100
    - Non_Worker_Pct         : NON_WORK_P / TOT_P * 100
    - Agri_Worker_Pct        : (MAIN_CL_P + MAIN_AL_P) / TOT_WORK_P * 100

    Division is guarded against zero denominators (result is NaN).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned district DataFrame (output of clean_district_data).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional derived indicator columns appended.
    """
    df = df.copy()

    def safe_div(num, den):
        """Divide two Series, returning NaN where denominator is zero."""
        return num.where(den != 0).div(den.where(den != 0))

    # Population aged 7+ (denominator for literacy rate)
    pop_7plus   = df["TOT_P"] - df.get("P_06", pd.Series(0, index=df.index))
    pop_f_7plus = df["TOT_F"] - df.get("F_06", pd.Series(0, index=df.index))
    pop_m_7plus = df["TOT_M"] - df.get("M_06", pd.Series(0, index=df.index))

    if "TOT_F" in df.columns and "TOT_M" in df.columns:
        df["Sex_Ratio"] = safe_div(df["TOT_F"], df["TOT_M"]) * 1000

    if "P_LIT" in df.columns:
        df["Literacy_Rate"] = safe_div(df["P_LIT"], pop_7plus) * 100

    if "F_LIT" in df.columns:
        df["Female_Literacy_Rate"] = safe_div(df["F_LIT"], pop_f_7plus) * 100

    if "M_LIT" in df.columns:
        male_lit_rate = safe_div(df["M_LIT"], pop_m_7plus) * 100
        if "Female_Literacy_Rate" in df.columns:
            df["Gender_Literacy_Gap"] = male_lit_rate - df["Female_Literacy_Rate"]

    if "TOT_WORK_P" in df.columns:
        df["Worker_Participation"] = safe_div(df["TOT_WORK_P"], df["TOT_P"]) * 100

    if "TOT_WORK_F" in df.columns and "TOT_F" in df.columns:
        df["Female_Worker_Part"] = safe_div(df["TOT_WORK_F"], df["TOT_F"]) * 100

    if "P_06" in df.columns:
        df["Child_Pop_Pct"] = safe_div(df["P_06"], df["TOT_P"]) * 100

    if "P_SC" in df.columns:
        df["SC_Pop_Pct"] = safe_div(df["P_SC"], df["TOT_P"]) * 100

    if "P_ST" in df.columns:
        df["ST_Pop_Pct"] = safe_div(df["P_ST"], df["TOT_P"]) * 100

    if "NON_WORK_P" in df.columns:
        df["Non_Worker_Pct"] = safe_div(df["NON_WORK_P"], df["TOT_P"]) * 100

    if all(c in df.columns for c in ["MAIN_CL_P", "MAIN_AL_P", "TOT_WORK_P"]):
        agri = df["MAIN_CL_P"] + df["MAIN_AL_P"]
        df["Agri_Worker_Pct"] = safe_div(agri, df["TOT_WORK_P"]) * 100

    return df


# ============================================================
# SECTION 7 — EXPLORATORY DATA ANALYSIS
# ============================================================
# Functions will be implemented in the EDA development phase.
# Planned analyses:
#   - Distribution of key indicators (histograms, box plots)
#   - State-wise aggregations and comparisons
#   - Top/bottom districts by indicator
#   - Correlation matrix of derived indicators
#   - Regional pattern identification

def run_eda(df: pd.DataFrame) -> dict:
    """
    Placeholder for Exploratory Data Analysis.

    To be implemented in the EDA development phase.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered district DataFrame.

    Returns
    -------
    dict
        Dictionary of EDA results / figures (to be populated).
    """
    # TODO: implement EDA in development phase
    return {}


# ============================================================
# SECTION 8 — STATISTICAL ANALYSIS
# ============================================================
# Planned:
#   - Descriptive statistics (mean, median, std, IQR)
#   - Correlation analysis
#   - Regional and state-level aggregates

def run_statistical_analysis(df: pd.DataFrame) -> dict:
    """
    Placeholder for Statistical Analysis.

    To be implemented in the statistical analysis development phase.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered district DataFrame.

    Returns
    -------
    dict
        Dictionary of statistical results (to be populated).
    """
    # TODO: implement statistical analysis in development phase
    return {}


# ============================================================
# SECTION 9 — MACHINE LEARNING: K-MEANS CLUSTERING
# ============================================================
# Planned:
#   - Select clustering features (derived indicators)
#   - Scale features with StandardScaler
#   - Determine optimal k via elbow method (inertia curve)
#   - Fit KMeans and assign cluster labels
#   - Describe clusters using centroid feature values

def run_kmeans(df: pd.DataFrame, features: list, n_clusters: int = N_CLUSTERS):
    """
    Placeholder for K-Means clustering.

    To be implemented in the ML development phase.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered district DataFrame.
    features : list of str
        Column names to use as clustering features.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Cluster' column appended.
    """
    # TODO: implement K-Means in ML development phase
    return df


# ============================================================
# SECTION 10 — PCA VISUALISATION
# ============================================================
# Planned:
#   - Reduce feature matrix to 2 / 3 principal components
#   - Plot district positions in PCA space coloured by cluster
#   - Display explained variance ratio
#   - Interpret principal component loadings

def run_pca(df: pd.DataFrame, features: list, n_components: int = 2):
    """
    Placeholder for Principal Component Analysis.

    To be implemented in the PCA development phase.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered district DataFrame.
    features : list of str
        Column names to reduce.
    n_components : int
        Number of components to retain.

    Returns
    -------
    pd.DataFrame
        DataFrame with PCA component columns appended.
    """
    # TODO: implement PCA in ML development phase
    return df


# ============================================================
# SECTION 11 — ANOMALY DETECTION
# ============================================================
# Planned:
#   - Apply Isolation Forest on scaled indicator features
#   - Flag anomalous districts for further investigation
#   - Visualise anomalies in PCA / indicator space

def run_anomaly_detection(df: pd.DataFrame, features: list,
                          contamination: float = CONTAMINATION):
    """
    Placeholder for Isolation Forest anomaly detection.

    To be implemented in the anomaly detection development phase.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered district DataFrame.
    features : list of str
        Column names to use for anomaly scoring.
    contamination : float
        Expected proportion of anomalies.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Anomaly' column (-1 = anomaly, 1 = normal).
    """
    # TODO: implement Isolation Forest in anomaly detection phase
    return df


# ============================================================
# SECTION 12 — AI-ASSISTED INSIGHTS
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
    """
    Placeholder for AI-assisted analytical storytelling.

    Will generate structured observations, insights, hypotheses,
    and recommendations from model and EDA results.

    To be implemented after ML and EDA sections are complete.

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed district DataFrame (with clusters, anomalies,
        and derived indicators).

    Returns
    -------
    list of dict
        Each dict contains: observation, insight, hypothesis,
        recommendation.
    """
    # TODO: implement insights generation in insights development phase
    return []


# ============================================================
# SECTION 13 — STREAMLIT DASHBOARD
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
                          validation: dict):
    """Render the Dataset Overview page."""
    st.title("📂 Dataset Overview")
    st.markdown("""
    **Source:** Primary Census Abstract (PCA), Census of India 2011  
    **Publisher:** Office of the Registrar General & Census Commissioner, India  
    **URL:** https://censusindia.gov.in/nada/index.php/catalog/6191
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows (raw)", f"{validation['total_rows']:,}")
    col2.metric("Total Columns", f"{validation['total_columns']}")
    col3.metric("District-Total Rows", f"{validation['district_total_rows']:,}")

    st.subheader("Geographic Levels in Dataset")
    st.write(validation["unique_levels"])

    st.subheader("TRU Values in Dataset")
    st.write(validation["unique_tru"])

    st.subheader("Filtered District Dataset (first 20 rows)")
    st.dataframe(district_df.head(20), use_container_width=True)

    st.subheader("Missing Values (raw dataset, top 20 columns)")
    missing = pd.Series(validation["missing_per_col"]).sort_values(ascending=False)
    st.dataframe(missing[missing > 0].head(20).rename("Missing Count"),
                 use_container_width=True)

    st.info(f"Duplicate rows in raw dataset: **{validation['duplicate_rows']}**")


def page_coming_soon(section_name: str):
    """Placeholder for sections not yet implemented."""
    st.title(f"🚧 {section_name}")
    st.info(
        f"**{section_name}** will be implemented in the next development phase.  \n"
        "Data loading and preprocessing are already operational — "
        "analytical sections are under active development."
    )


# ============================================================
# SECTION 14 — MAIN APPLICATION
# ============================================================

def main():
    """
    Entry point for the Streamlit application.

    Loads data, validates it, engineers features, and renders
    the selected dashboard section.
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

    # ── Data pipeline (cached) ─────────────────────────────────
    @st.cache_data(show_spinner="Loading dataset…")
    def load_pipeline():
        raw      = load_raw_data()
        valid    = validate_data(raw)
        district = get_district_data(raw)
        cleaned  = clean_district_data(district)
        featured = engineer_features(cleaned)
        return raw, valid, cleaned, featured

    try:
        raw_df, validation, district_df, featured_df = load_pipeline()
    except FileNotFoundError:
        st.error(
            f"Dataset not found at `{DATA_PATH}`.  \n"
            "Please ensure the file exists at:  \n"
            "`data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx`"
        )
        st.stop()

    # ── Page routing ───────────────────────────────────────────
    if selection == "Home":
        page_home()
    elif selection == "Dataset Overview":
        page_dataset_overview(raw_df, district_df, validation)
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
# SECTION 15 — ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
