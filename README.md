# IndiDevAI
### AI-Powered District Development Intelligence for India

[![IBM SkillsBuild](https://img.shields.io/badge/IBM%20SkillsBuild-Internship%202026-0062FF?style=flat-square)](https://skillsbuild.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)

---

## Project Overview

**IndiDevAI** is a data analytics and AI/ML project that analyses Indian district-level
demographic, education, and employment characteristics using the **Census of India 2011**
Primary Census Abstract (PCA) dataset.

The project delivers an interactive Streamlit dashboard with AI-assisted analytical
storytelling, machine learning–based district profiling, and data-driven recommendations.

---

## Problem Statement

India's 640+ districts exhibit significant socioeconomic variation that is not captured
by state-level or national aggregates alone. Identifying patterns, disparities, and
anomalies at the district level is essential for targeted policy design and resource
allocation. This project applies data analytics and machine learning to the Census 2011
PCA data to surface these district-level patterns.

---

## Objectives

1. Perform end-to-end data cleaning and feature engineering on Census 2011 PCA data.
2. Conduct exploratory data analysis to identify distributional patterns and regional disparities.
3. Engineer meaningful socioeconomic indicators (literacy rate, sex ratio, worker participation, etc.).
4. Apply K-Means clustering to segment districts into socioeconomic profiles.
5. Use Principal Component Analysis (PCA) to visualise district positioning in reduced-dimension space.
6. Detect anomalous districts using Isolation Forest.
7. Generate structured AI-assisted insights following the Observations → Insights → Hypotheses → Recommendations framework.
8. Present findings through an interactive Streamlit dashboard.

---

## Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | Primary Census Abstract (PCA) — India & States/UTs, State and District Level |
| **Year** | Census of India 2011 |
| **Publisher** | Office of the Registrar General & Census Commissioner, India |
| **Official Source** | https://censusindia.gov.in/nada/index.php/catalog/6191 |
| **File** | `data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx` |
| **Sheet** | Sheet1 |
| **Total Rows** | 2 028 (excluding header) |
| **Total Columns** | 94 |
| **Primary Analysis Scope** | `Level == "DISTRICT"` AND `TRU == "Total"` |

### Key Column Groups

| Prefix / Column | Description |
|-----------------|-------------|
| `State`, `District` | State and district code identifiers |
| `Level` | Geographic level (India / STATE / DISTRICT) |
| `Name` | Geographic entity name |
| `TRU` | Total / Rural / Urban |
| `TOT_P/M/F` | Total / Male / Female population |
| `P_06`, `M_06`, `F_06` | Population aged 0–6 |
| `P_SC`, `P_ST` | Scheduled Caste / Scheduled Tribe population |
| `P_LIT`, `M_LIT`, `F_LIT` | Literate population (Total / Male / Female) |
| `TOT_WORK_P/M/F` | Total workers |
| `MAINWORK_P/M/F` | Main workers |
| `MAIN_CL_P` | Cultivators (main) |
| `MAIN_AL_P` | Agricultural labourers (main) |
| `MAIN_HH_P` | Household industry workers (main) |
| `MAIN_OT_P` | Other workers (main) |
| `MARGWORK_P/M/F` | Marginal workers |
| `NON_WORK_P/M/F` | Non-workers |

---

## Data Processing Methodology

### Raw Census Variables

The following columns are read directly from the Excel dataset without modification:

| Group | Raw Columns |
|-------|-------------|
| Identifiers | `State`, `District`, `Subdistt`, `Level`, `Name`, `TRU` |
| Households | `No_HH` |
| Total Population | `TOT_P`, `TOT_M`, `TOT_F` |
| Child Population (0–6) | `P_06`, `M_06`, `F_06` |
| Scheduled Caste | `P_SC`, `M_SC`, `F_SC` |
| Scheduled Tribe | `P_ST`, `M_ST`, `F_ST` |
| Literate Population | `P_LIT`, `M_LIT`, `F_LIT` |
| Illiterate Population | `P_ILL`, `M_ILL`, `F_ILL` |
| Total Workers | `TOT_WORK_P`, `TOT_WORK_M`, `TOT_WORK_F` |
| Main Workers | `MAINWORK_P`, `MAINWORK_M`, `MAINWORK_F` |
| Main Cultivators | `MAIN_CL_P`, `MAIN_CL_M`, `MAIN_CL_F` |
| Main Agricultural Labourers | `MAIN_AL_P`, `MAIN_AL_M`, `MAIN_AL_F` |
| Main HH Industry | `MAIN_HH_P`, `MAIN_HH_M`, `MAIN_HH_F` |
| Main Other Workers | `MAIN_OT_P`, `MAIN_OT_M`, `MAIN_OT_F` |
| Marginal Workers | `MARGWORK_P`, `MARGWORK_M`, `MARGWORK_F` |
| Marginal Cultivators | `MARG_CL_P`, `MARG_CL_M`, `MARG_CL_F` |
| Marginal Agricultural Labourers | `MARG_AL_P`, `MARG_AL_M`, `MARG_AL_F` |
| Marginal HH Industry | `MARG_HH_P`, `MARG_HH_M`, `MARG_HH_F` |
| Marginal Other Workers | `MARG_OT_P`, `MARG_OT_M`, `MARG_OT_F` |
| Non-Workers | `NON_WORK_P`, `NON_WORK_M`, `NON_WORK_F` |

### Derived Project Features

All derived features are computed in [`engineer_features()`](Aarav_IndiDevAI.py) using zero-denominator-guarded division. Every formula was verified against the Census of India 2011 definition before implementation.

**Demographic Features**

| Feature | Formula | Source Columns | Interpretation |
|---------|---------|----------------|----------------|
| `Sex_Ratio` | `(TOT_F / TOT_M) × 1000` | `TOT_F`, `TOT_M` | Females per 1,000 males; lower values indicate gender imbalance |
| `Child_Pop_Pct` | `(P_06 / TOT_P) × 100` | `P_06`, `TOT_P` | Share of population aged 0–6; proxy for birth rate and child dependency |
| `SC_Pop_Pct` | `(P_SC / TOT_P) × 100` | `P_SC`, `TOT_P` | Scheduled Caste population as share of total population |
| `ST_Pop_Pct` | `(P_ST / TOT_P) × 100` | `P_ST`, `TOT_P` | Scheduled Tribe population as share of total population |

**Education Features**

> **Denominator note:** The Census of India defines the effective literacy rate as literate persons as a share of the population aged **7 and above** (children aged 0–6 are excluded because literacy is not enumerated for them). All literacy-rate denominators therefore subtract the 0–6 age-group column from the corresponding total population.

| Feature | Formula | Source Columns | Interpretation |
|---------|---------|----------------|----------------|
| `Literacy_Rate` | `(P_LIT / (TOT_P − P_06)) × 100` | `P_LIT`, `TOT_P`, `P_06` | Effective literacy rate for population aged 7+ |
| `Male_Literacy_Rate` | `(M_LIT / (TOT_M − M_06)) × 100` | `M_LIT`, `TOT_M`, `M_06` | Male effective literacy rate |
| `Female_Literacy_Rate` | `(F_LIT / (TOT_F − F_06)) × 100` | `F_LIT`, `TOT_F`, `F_06` | Female effective literacy rate |
| `Gender_Literacy_Gap` | `Male_Literacy_Rate − Female_Literacy_Rate` | derived | Percentage-point gap; positive = male advantage; negative = female advantage (e.g. Jaintia Hills, Meghalaya) |

**Employment Features**

| Feature | Formula | Source Columns | Interpretation |
|---------|---------|----------------|----------------|
| `Worker_Participation` | `(TOT_WORK_P / TOT_P) × 100` | `TOT_WORK_P`, `TOT_P` | Overall work participation rate (total workers as % of total population) |
| `Female_Worker_Part` | `(TOT_WORK_F / TOT_F) × 100` | `TOT_WORK_F`, `TOT_F` | Female work participation rate |
| `Main_Worker_Pct` | `(MAINWORK_P / TOT_WORK_P) × 100` | `MAINWORK_P`, `TOT_WORK_P` | Main workers as % of total workers (worked 6+ months/year) |
| `Marginal_Worker_Pct` | `(MARGWORK_P / TOT_WORK_P) × 100` | `MARGWORK_P`, `TOT_WORK_P` | Marginal workers as % of total workers (worked <6 months/year) |
| `Non_Worker_Pct` | `(NON_WORK_P / TOT_P) × 100` | `NON_WORK_P`, `TOT_P` | Non-workers as % of total population |
| `Agri_Worker_Pct` | `((MAIN_CL_P + MAIN_AL_P) / TOT_WORK_P) × 100` | `MAIN_CL_P`, `MAIN_AL_P`, `TOT_WORK_P` | Combined main cultivators and main agricultural labourers as % of total workers |

### Cleaning Operations Applied

| Operation | Justification |
|-----------|---------------|
| Strip whitespace from string columns | Prevents mismatches in Level/TRU filtering |
| Drop entirely-empty columns | 6 trailing placeholder columns (CQ–CV) contain no data |
| Convert numeric Census columns from string to float | Raw Excel loaded with `dtype=str` to preserve zero-padded codes; explicit conversion required |
| Drop districts with zero/missing `TOT_P` | Rates cannot be computed for zero-population records |

**What was intentionally NOT done:**
- Missing values were **not** filled with zero (NaN may represent genuinely absent populations)
- No values were silently clipped or capped
- Negative `Gender_Literacy_Gap` values (1 district: Jaintia Hills, Meghalaya) were retained — this is a real documented pattern in matrilineal societies, not a data error

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Data processing | pandas, numpy |
| Visualisation | plotly, matplotlib, seaborn |
| Machine learning | scikit-learn (KMeans, PCA, IsolationForest, StandardScaler) |
| Dashboard | Streamlit |
| Excel I/O | openpyxl |

---

## Methodology

```
Raw Excel Data
    │
    ▼
Data Loading (pandas read_excel)
    │
    ▼
Data Validation (row counts, Level/TRU uniques, missing values)
    │
    ▼
Filtering (Level == DISTRICT, TRU == Total)
    │
    ▼
Data Cleaning (strip whitespace, drop zero-population rows, type casting)
    │
    ▼
Feature Engineering (derived indicators: literacy rate, sex ratio, etc.)
    │
    ├──▶ EDA (distributions, correlations, state aggregates, top/bottom districts)
    │
    ├──▶ Statistical Analysis (descriptive stats, regional aggregates)
    │
    ├──▶ K-Means Clustering (district socioeconomic segmentation)
    │
    ├──▶ PCA (dimensionality reduction, district profiling)
    │
    ├──▶ Isolation Forest (anomaly detection)
    │
    └──▶ AI-Assisted Insights (Observations → Insights → Hypotheses → Recommendations)
                │
                ▼
        Streamlit Dashboard
```

---

## ML Techniques

| Technique | Library | Purpose |
|-----------|---------|---------|
| **K-Means Clustering** | scikit-learn | Group districts with similar socioeconomic profiles |
| **Principal Component Analysis (PCA)** | scikit-learn | Reduce 11 dimensions to 2; visualise district positioning |
| **Isolation Forest** | scikit-learn | Detect districts with unusual indicator combinations |

> **Scope note:** All ML is unsupervised and exploratory. The focus is on district
> profiling and pattern discovery using historical 2011 data. No future predictions
> are made.

### ML Feature Shortlist (11 indicators)

| Feature | Description |
|---------|-------------|
| `Sex_Ratio` | Females per 1,000 males |
| `Child_Pop_Pct` | Population aged 0–6 as % of total |
| `SC_Pop_Pct` | Scheduled Caste population % |
| `ST_Pop_Pct` | Scheduled Tribe population % |
| `Literacy_Rate` | Overall literacy rate (pop 7+) |
| `Female_Literacy_Rate` | Female literacy rate (pop 7+) |
| `Gender_Literacy_Gap` | Male literacy rate − Female literacy rate (pp) |
| `Worker_Participation` | Total workers as % of total population |
| `Female_Worker_Part` | Female workers as % of female population |
| `Agri_Worker_Pct` | (Cultivators + Agricultural labourers) as % of total workers |
| `Main_Worker_Pct` | Main workers as % of total workers |

**Excluded features:** `Male_Literacy_Rate` (collinear), `Non_Worker_Pct` (arithmetic complement of Worker_Participation), `Marginal_Worker_Pct` (complement of Main_Worker_Pct).

---

## Streamlit Application Sections

| # | Section | Status |
|---|---------|--------|
| 1 | Home / Project Overview | ✅ Implemented |
| 2 | Dataset Overview | ✅ Implemented |
| 3 | Demographics | ✅ Implemented |
| 4 | Education | ✅ Implemented |
| 5 | Employment | ✅ Implemented |
| 6 | Exploratory Analysis | ✅ Implemented |
| 7 | District Clustering | ✅ Implemented |
| 8 | PCA Visualisation | ✅ Implemented |
| 9 | Anomaly Detection | ✅ Implemented |
| 10 | AI-Assisted Insights | ✅ Implemented |
| 11 | Recommendations | 🔲 Planned |
| 12 | Methodology / About | 🔲 Planned |

---

## Project Structure

```
IndiDevAI/
│
├── data/
│   ├── raw/
│   │   └── DDW_PCA0000_2011_Indiastatedist.xlsx   ← Original Census data (read-only)
│   └── processed/
│       ├── district_analysis_ready.csv            ← 640 districts × 75 columns (pipeline output)
│       └── district_ml_results.csv               ← 640 districts × 20 columns (ML output)
│
├── Aarav_IndiDevAI.py       ← Single project code file (Streamlit app)
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

> **Important:** The entire project is contained in **one Python file** (`Aarav_IndiDevAI.py`)
> as required by the IBM SkillsBuild Academic Internship 2026 submission guidelines.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/IndiDevAI.git
cd IndiDevAI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run Aarav_IndiDevAI.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Project Status

**Phase 1 — Project Initialisation: ✅ Complete**

- Repository structure established
- Dataset inspected and confirmed: 2,028 rows × 94 columns, 640 DISTRICT/Total records
- `Aarav_IndiDevAI.py` created with full section scaffold

**Phase 2 — Data Validation, Cleaning & Feature Engineering: ✅ Complete**

- `load_dataset()`, `validate_dataset()` — 8-point quality check
- `clean_dataset()`, `filter_district_data()` — 640 district records, 0 duplicates
- `engineer_features()` — 14 derived socioeconomic indicators (Census-correct denominators)
- `validate_features()` — NaN / Inf / out-of-range checks
- Output: `data/processed/district_analysis_ready.csv` (640 rows × 75 columns)
- Notable: Jaintia Hills (Meghalaya) has negative Gender_Literacy_Gap (female literacy exceeds male) — retained as valid

**Phase 3 — EDA: ✅ Complete**

- Descriptive statistics across demographics, education, employment
- Distribution and outlier analysis (IQR-based)
- Correlation matrix (Pearson) across all derived features
- State-level aggregates; top/bottom district rankings
- 57 programmatic observations generated from data
- Key finding: Literacy Rate range 36.10% (Alirajpur) – 97.91% (Serchhip)
- Key finding: Child_Pop_Pct vs Literacy_Rate correlation r = −0.678
- Streamlit pages: Demographics, Education, Employment, Exploratory Analysis

**Phase 4 — AI-Assisted Analytical Storytelling: ✅ Complete**

- Framework: Observations → Insights → Hypotheses → Recommendations
- 7 observations, 6 insights, 3 explicitly-labelled hypotheses, 3 recommendations
- Automated validation (28 checks) — 0 causal-language violations, 0 fabricated values
- Language discipline enforced: "is associated with", "may indicate", "suggests"
- Streamlit page: AI-Assisted Insights

**Phase 5 — Machine Learning: ✅ Complete**

- `prepare_ml_data()` — StandardScaler, variance checks, 11-feature shortlist
- `run_kmeans()` — K=2..8 silhouette evaluation; **K=4 selected** (silhouette=0.2232)
- `run_pca_analysis()` — full PCA; **PC1=34.4%, PC2=24.3%** (58.7% combined)
- `run_anomaly_detection()` — Isolation Forest (200 estimators); **32 unusual profiles** (5.0%)
- `validate_ml_pipeline()` — 12 checks, all passed
- Output: `data/processed/district_ml_results.csv` (640 rows × 20 columns)
- Streamlit pages: District Clustering, PCA Visualisation, Anomaly Detection (unified multi-tab page)

**Next:** Recommendations page, Methodology / About page, Project Report (.docx)

---

## Internship Submission Files

| File | Description |
|------|-------------|
| `Aarav_IndiDevAI.py` | Complete project code (single file) |
| `requirements.txt` | Python dependencies |
| `Aarav_IndiDevAI_ProjectReport.docx` | Project report (to be created) |
| `README.md` | This file |

---

## Important Notes

- Census 2011 is a **historical snapshot**. No future predictions are made.
- All analytical language follows careful phrasing: "is associated with", "may indicate", "suggests a possible relationship".
- All results are derived from actual data — no fabricated statistics or model outputs.

---

*IBM SkillsBuild Academic Internship 2026 — Data Analytics with AI*
