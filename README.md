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

## Planned Methodology

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

## Planned ML Techniques

| Technique | Purpose |
|-----------|---------|
| **K-Means Clustering** | Group districts with similar socioeconomic profiles |
| **Principal Component Analysis (PCA)** | Reduce dimensionality; visualise district positioning |
| **Isolation Forest** | Detect districts with unusual indicator patterns |

> **Scope note:** The ML focus is on district profiling and pattern discovery using
> historical 2011 data. No future predictions are made.

---

## Streamlit Application Sections

| # | Section | Status |
|---|---------|--------|
| 1 | Home / Project Overview | ✅ Implemented |
| 2 | Dataset Overview | ✅ Implemented |
| 3 | Demographics | 🔲 Planned |
| 4 | Education | 🔲 Planned |
| 5 | Employment | 🔲 Planned |
| 6 | District Comparison | 🔲 Planned |
| 7 | Exploratory Analysis | 🔲 Planned |
| 8 | District Clustering | 🔲 Planned |
| 9 | PCA Visualisation | 🔲 Planned |
| 10 | Anomaly Detection | 🔲 Planned |
| 11 | AI-Assisted Insights | 🔲 Planned |
| 12 | Recommendations | 🔲 Planned |
| 13 | Methodology / About | 🔲 Planned |

---

## Project Structure

```
IndiDevAI/
│
├── data/
│   └── raw/
│       └── DDW_PCA0000_2011_Indiastatedist.xlsx
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
- Dataset inspected and moved to `data/raw/`
- `Aarav_IndiDevAI.py` created with full section scaffold
- Data loading, validation, cleaning, and feature engineering functions implemented
- Streamlit Home and Dataset Overview pages operational
- Remaining dashboard sections are scaffolded with clear placeholders

Next phase: Demographics, Education, and Employment analytical sections.

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
