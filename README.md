# Teff Maturity Date Classifier

A Streamlit web app that predicts whether a teff accession matures **Early**, **Medium**, or **Late**
using a Random Forest classifier trained on Ethiopian teff germplasm data.

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the model artifacts
Open and run **all cells** in `teff_md_classification_fixed (2).ipynb`
(Google Colab or locally with Jupyter).  
This produces two files that the app needs:
- `teff_pipeline.joblib` — the trained inference pipeline
- `teff_insights.joblib` — metrics and charts for the Insights page

Place both files in the same directory as `streamlit_app.py`.

### 3. Run the app
```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## Pages

| Page | Description |
|---|---|
| Single Prediction | Enter agronomic + bioclimatic values and get an instant prediction with agronomic advice |
| Batch Prediction (CSV) | Upload a CSV of multiple accessions and download results |
| Model Insights | Confusion matrix, permutation importance, and key metrics |

## Required input columns (raw units)

| Column | Description | Example |
|---|---|---|
| `GfilP` | Grain filling period (days) | 49 |
| `Altitude` | Site altitude (m) | 2100 |
| `ph_kcl_60cm` | Soil pH (KCl, 60 cm) | 5.4 |
| `bio12_1970_2000` | Annual precipitation (mm) | 1060 |
| `bio5_1970_2000` | Max temp of warmest month (°C) | 27.3 |
| `bio16_1970_2000` | Precipitation of wettest quarter (mm) | 625 |
| `Hdatec` | Heading date category | Early / Medium / Late |
| `Ptype` | Panicle type | very loose / loose / semi-loose / compact / very compact |

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing to `streamlit_app.py`.
3. Upload `teff_pipeline.joblib` and `teff_insights.joblib` to the repo (or use `st.secrets` + cloud storage).
