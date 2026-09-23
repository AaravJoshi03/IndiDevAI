# IndiDevAI
## AI-Powered District Development Intelligence for India

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)

## 1. Project Overview

IndiDevAI is a data analytics and machine learning project that analyses Indian district-level demographic, education, and employment characteristics using the Census of India 2011 Primary Census Abstract (PCA) dataset.

The project encompasses:
- District-level socioeconomic analysis
- Data validation and cleaning
- Feature engineering
- Exploratory data analysis
- AI-assisted analytical storytelling
- Unsupervised machine learning
- Interactive Streamlit dashboard

The analytical scope comprises 640 district-level records across 35 states and Union Territories. All analyses are based on historical Census of India 2011 data; the project does not make future prediction claims.

## 2. Dataset

| Attribute | Details |
|-----------|---------|
| Name | Primary Census Abstract (PCA) — India & States/UTs, State and District Level |
| Year | Census of India 2011 |
| Publisher | Office of the Registrar General & Census Commissioner, India |
| Official Source | https://censusindia.gov.in/nada/index.php/catalog/6191 |
| File | `data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx` |
| Total Rows | 2,028 (excluding header) |
| Total Columns | 94 |

### Analytical Scope

- 640 district-level analytical records
- 35 states and Union Territories
- `Level == "DISTRICT"`
- `TRU == "Total"`

The analysis uses the district-total population records (`Level == "DISTRICT"` and `TRU == "Total"`) to examine aggregate district-level indicators.

## 3. Key Analytical Components

| Component | Description |
|-----------|-------------|
| Data Validation & Cleaning | Quality checks, cleaning, type conversion and validation of source data |
| Feature Engineering | 14 derived socioeconomic indicators |
| Exploratory Data Analysis | Distribution analysis, correlations, state-level aggregates and district comparisons |
| Analytical Storytelling | Observations → Insights → Hypotheses → Recommendations |
| K-Means Clustering | Groups districts based on selected socioeconomic indicators |
| Principal Component Analysis | Reduces dimensionality and visualises district positioning |
| Isolation Forest | Identifies districts with unusual indicator combinations |
| District Explorer | Allows district-level inspection and comparison against dataset averages |

### Feature Engineering

The project derives 14 district-level indicators covering demographic, education and employment characteristics, including literacy rates, sex ratio, child population share, worker participation and agricultural worker share. The complete feature definitions and formulas are documented in the project report.

## 4. Technologies Used

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| Data Processing | pandas, NumPy |
| Statistical Analysis | SciPy |
| Machine Learning | scikit-learn |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Excel I/O | OpenPyXL |

## 5. Machine Learning

The machine learning component of IndiDevAI is unsupervised and exploratory, designed for district profiling and pattern discovery using historical 2011 Census data. No future predictions are made.

| Technique | Purpose |
|-----------|---------|
| K-Means Clustering | Group districts with similar socioeconomic profiles |
| Principal Component Analysis (PCA) | Reduce selected features to principal components for visualisation |
| Isolation Forest | Identify districts with unusual indicator combinations |

### Final Verified Results

| Result | Value |
|--------|-------|
| Optimal K | 4 clusters |
| Silhouette Score | 0.2232 |
| PCA — PC1 variance explained | 34.4% |
| PCA — PC2 variance explained | 24.3% |
| PCA — PC1 + PC2 combined | 58.7% |
| Unusual district profiles | 32 (5.0% of 640) |

## 6. Streamlit Application

The Streamlit dashboard is structured around the following top-level navigation sections:

### Overview
1. Home
2. Dataset Overview

### Analysis
3. Demographics
4. Education
5. Employment
6. Exploratory Analysis
7. AI Insights

### Machine Learning
8. Machine Learning
9. ML Interpretation
10. Recommendations

### Documentation
11. Methodology

The Machine Learning page contains the ML analysis and interactive tabs for clustering, PCA, and anomaly detection.

## 7. Project Structure

```text
IndiDevAI/
├── data/
│   ├── raw/
│   │   └── DDW_PCA0000_2011_Indiastatedist.xlsx
│   └── processed/
│       ├── district_analysis_ready.csv
│       └── district_ml_results.csv
├── AaravJoshi_IndiDevAI.py
├── AaravJoshi_ProjectReport.docx
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

AaravJoshi_IndiDevAI.py contains the complete Streamlit application and project logic in a single Python file.

## 8. Setup and Run Instructions

### Clone the repository
```bash
git clone https://github.com/AaravJoshi03/IndiDevAI.git
cd IndiDevAI
```

### Create a virtual environment
```bash
python -m venv .venv
```

### Activate on Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
python -m streamlit run AaravJoshi_IndiDevAI.py
```

The application will be available at:
http://localhost:8501

## 9. Key Results

| Metric | Value |
|--------|-------|
| Districts analysed | 640 |
| States and Union Territories | 35 |
| Derived indicators engineered | 14 |
| K-Means clusters | 4 |
| K-Means silhouette score | 0.2232 |
| PCA variance explained (PC1 + PC2) | 58.7% |
| Unusual district profiles identified | 32 |

### Notable EDA Findings

- Literacy Rate range: 36.10% to 97.91%
- Child Population Share vs Literacy Rate: r = -0.678
- Female Literacy Rate minimum: 30.29%
- Maximum Gender Literacy Gap: 34.0 percentage points
- Agri Worker Share vs Literacy Rate: r = -0.399
- One district, Jaintia Hills, Meghalaya, records a negative Gender_Literacy_Gap, meaning female literacy exceeds male literacy in that district. The value was retained as a valid observation.

## 10. Limitations

- The Census of India 2011 is a historical dataset.
- Results reflect conditions recorded in 2011 and should not be interpreted as current 2026 district conditions.
- Correlation does not imply causation.
- K-Means clusters are data-derived and exploratory.
- Cluster labels should not be interpreted as normative judgments about districts.
- Isolation Forest anomalies indicate statistical unusualness relative to this dataset and require contextual investigation.
- Additional and more current datasets would be required for contemporary analysis.

## 11. Future Scope

- Integration of more recent census or administrative datasets for temporal comparisons.
- Additional socioeconomic indicators such as health, infrastructure and income proxies.
- Comparative analysis using datasets such as NFHS or NSSO where appropriate.
- Supervised learning approaches if labelled outcome data becomes available.

## 12. License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 AaravJoshi03
