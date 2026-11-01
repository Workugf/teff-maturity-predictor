import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Teff Maturity Predictor", layout="wide")

# ── Background image (CC-BY-SA 3.0 Rasbak / Wikimedia Commons) ────────────────
st.markdown("""
<style>
/* Full-page background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/3/3e/Teff_pluim_Eragrostis_tef.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
/* Dark overlay so text stays readable */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.55);
    z-index: 0;
}
/* Keep all content above the overlay */
[data-testid="stAppViewContainer"] > * {
    position: relative;
    z-index: 1;
}
/* Translucent card behind the main block */
[data-testid="stVerticalBlock"] {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem;
}
/* Make header text white */
h1, h2, h3, h4, p, label, div[data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}
/* Lighten tab labels */
[data-testid="stTabs"] button[role="tab"] {
    color: #e0e0e0 !important;
    font-weight: 600;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #4CAF50;
}
/* Sidebar stays clean if shown */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.6);
}
</style>
""", unsafe_allow_html=True)

# ---------- must be identical to the notebook ----------
NUM_FEATURES = ['GfilP', 'Altitude', 'ph_kcl_60cm', 'bio12_1970_2000',
                'Thermal_Moisture_Index', 'Growth_Season_Moisture_Index']
CAT_FEATURES = ['Hdatec', 'Ptype']
FEATURES = NUM_FEATURES + CAT_FEATURES
HDATEC_OPTIONS = ['Early', 'Medium', 'Late']
PTYPE_OPTIONS = ['very loose', 'loose', 'semi-loose', 'compact', 'very compact']
RAW_COLUMNS = ['GfilP', 'Altitude', 'ph_kcl_60cm', 'bio12_1970_2000',
               'bio5_1970_2000', 'bio16_1970_2000', 'Hdatec', 'Ptype']
CLASS_NAMES = ['Early', 'Medium', 'Late']


def add_engineered_features(frame):
    out = frame.copy()
    out['Thermal_Moisture_Index'] = out['bio5_1970_2000'] / (out['bio12_1970_2000'] + 1)
    out['Growth_Season_Moisture_Index'] = out['bio16_1970_2000'] / (out['bio12_1970_2000'] + 1)
    return out


@st.cache_resource
def load_artifacts():
    model = joblib.load('teff_pipeline.joblib')
    try:
        insights = joblib.load('teff_insights.joblib')
    except Exception:
        insights = None
    return model, insights


try:
    model, insights = load_artifacts()
except Exception as e:
    st.error(f"Could not load 'teff_pipeline.joblib': {e}")
    st.info("Run the notebook first to generate 'teff_pipeline.joblib', then place it in the same folder as this app.")
    st.stop()

ADVICE = {
    'Early': ("warning", "🌾", """
**Early maturing teff (drought escape)**
* **Context:** suits short rainy seasons (Belg), late sowing and drought-prone zones.
* **Water and climate:** good where rain is erratic or ends early, to escape terminal drought.
* **Fertilizer:** short cycle, so apply the second nitrogen/urea dose earlier, about 15–20 days after sowing.
* **Harvest:** prone to shattering; harvest as soon as stems and panicles turn straw-yellow.
* **Weeds and pests:** control weeds early, because the canopy closes quickly."""),
    'Medium': ("success", "🌾", """
**Medium maturing teff (standard main season)**
* **Context:** standard Meher-season crop (for example Quncho).
* **Water and climate:** best for mid to highland areas with a reliable 3–4 month rainy period.
* **Fertilizer:** NPS at planting, then urea top-dressing about 30 days after sowing.
* **Weeds and pests:** at least two hand-weeding rounds; watch for Epilachna beetle and shoot fly.
* **Harvest:** when about 90% of the field shows golden-yellow colour."""),
    'Late': ("info", "🌾", """
**Late maturing teff (needs long moisture)**
* **Context:** high-yield potential in long-rain highlands or with supplementary irrigation.
* **Water:** needs extended moisture; use tied ridges or irrigation if rain stops early.
* **Lodging:** taller plants lodge easily; use the recommended seed rate and balanced potassium.
* **Fertilizer:** split urea into two top-dressings (about day 30 and day 50–60).
* **Pests and disease:** longer field exposure; scout regularly for rust and seed-head insects."""),
}


def predict_frame(raw_df):
    missing = [c for c in RAW_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    raw_df = raw_df.copy()
    for c in ['Hdatec', 'Ptype']:
        raw_df[c] = raw_df[c].astype(str).str.strip()
    X = add_engineered_features(raw_df[RAW_COLUMNS])[FEATURES]
    return model.predict_proba(X)


st.title("🌾 Teff Maturity Date Predictor")
st.write(
    "Predict whether a teff accession matures **Early**, **Medium** or **Late**. "
    "Enter values in real units."
)

@st.cache_data
def load_eda_data():
    """Load and clean the raw dataset for EDA charts."""
    df = pd.read_excel('teff.xlsx')
    ptype_map  = {1:'very loose',2:'loose',3:'semi-loose',4:'compact',5:'very compact',
                  '1':'very loose','2':'loose','3':'semi-loose','4':'compact','5':'very compact'}
    hdatec_map = {1:'Early',2:'Medium',3:'Late','1':'Early','2':'Medium','3':'Late'}
    mdatec_map = {1:'Early',2:'Medium',3:'Late','1':'Early','2':'Medium','3':'Late'}
    curated = ['Hdatec','GfilP','Ptype','Altitude','ph_kcl_60cm',
               'bio12_1970_2000','bio5_1970_2000','bio16_1970_2000','Mdatec']
    existing = [c for c in curated if c in df.columns]
    df = df[existing].copy()
    for col in existing:
        df = df[df[col] != 86]
        df = df[df[col] != '86']
    df['Ptype']  = df['Ptype'].astype(str).str.strip().map(ptype_map)
    df['Hdatec'] = df['Hdatec'].map(hdatec_map).fillna(df['Hdatec'])
    df['Mdatec'] = df['Mdatec'].map(mdatec_map).fillna(df['Mdatec'])
    df = df.dropna(subset=['Ptype', 'Mdatec'])
    return df

tab1, tab2, tab3, tab4 = st.tabs(
    ["Single Prediction", "Batch Prediction (CSV)", "Model Insights", "Data Explorer"]
)

# ── Single Prediction ──────────────────────────────────────────────────────────
with tab1:
    st.header("Predict for a single sample")
    c1, c2 = st.columns(2)
    with c1:
        gfilp    = st.number_input("Grain filling period (days)", 30.0, 80.0, 49.0, 1.0)
        altitude = st.number_input("Altitude (m)", 1000.0, 3200.0, 2100.0, 10.0)
        ph_kcl   = st.number_input("Soil pH (KCl, 60 cm)", 4.0, 7.0, 5.4, 0.1)
        bio12    = st.number_input("Annual precipitation, bio12 (mm)", 500.0, 2000.0, 1060.0, 10.0)
    with c2:
        hdatec = st.selectbox("Heading date category", HDATEC_OPTIONS, index=1)
        ptype  = st.selectbox("Panicle type", PTYPE_OPTIONS, index=1)
        bio5   = st.number_input("Max temperature of warmest month, bio5 (°C)", 15.0, 40.0, 27.3, 0.1)
        bio16  = st.number_input("Precipitation of wettest quarter, bio16 (mm)", 200.0, 1200.0, 625.0, 10.0)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            'GfilP': gfilp, 'Altitude': altitude, 'ph_kcl_60cm': ph_kcl,
            'bio12_1970_2000': bio12, 'bio5_1970_2000': bio5,
            'bio16_1970_2000': bio16, 'Hdatec': hdatec, 'Ptype': ptype
        }])
        proba = predict_frame(row)[0]
        label = CLASS_NAMES[int(np.argmax(proba))]
        kind, emoji, text = ADVICE[label]
        getattr(st, kind)(f"Predicted maturity: **{emoji} {label}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Early",  f"{proba[0] * 100:.1f}%")
        m2.metric("Medium", f"{proba[1] * 100:.1f}%")
        m3.metric("Late",   f"{proba[2] * 100:.1f}%")
        st.markdown(text)

# ── Batch Prediction ───────────────────────────────────────────────────────────
with tab2:
    st.header("Batch prediction from CSV")
    st.write(
        "Required columns: " + ", ".join(f"`{c}`" for c in RAW_COLUMNS) +
        ".  `Hdatec`: Early / Medium / Late.  "
        "`Ptype`: very loose / loose / semi-loose / compact / very compact."
    )
    template = pd.DataFrame([{
        'GfilP': 49, 'Altitude': 2100, 'ph_kcl_60cm': 5.4, 'bio12_1970_2000': 1060,
        'bio5_1970_2000': 27.3, 'bio16_1970_2000': 625,
        'Hdatec': 'Medium', 'Ptype': 'loose'
    }])
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode('utf-8'),
        "teff_template.csv", "text/csv"
    )
    up = st.file_uploader("Choose a CSV file", type="csv")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            proba = predict_frame(df_in)
            out = df_in.copy()
            out['Predicted_Maturity'] = [CLASS_NAMES[i] for i in proba.argmax(axis=1)]
            out['Confidence_%'] = (proba.max(axis=1) * 100).round(1)
            st.dataframe(out)
            st.download_button(
                "Download predictions",
                out.to_csv(index=False).encode('utf-8'),
                "teff_predictions.csv", "text/csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ── Model Insights ─────────────────────────────────────────────────────────────
with tab3:
    st.header("Model insights")
    if insights is None:
        st.info("teff_insights.joblib not found — run the notebook to generate it.")
    else:
        mt = insights['metrics']
        a, b, c = st.columns(3)
        a.metric("Test accuracy",           f"{mt['test_accuracy']:.3f}")
        b.metric("Test macro-F1",           f"{mt['test_macro_f1']:.3f}")
        c.metric("CV macro-F1 (leak-free)", f"{mt['cv_macro_f1']:.3f}")

        left, right = st.columns(2)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            insights['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
        )
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion matrix (test set)')
        left.pyplot(fig)

        imp = insights['feature_importance'].sort_values('Importance')
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.barh(imp['Feature'], imp['Importance'], color='teal')
        ax2.set_title('Feature importance (Random Forest)')
        ax2.set_xlabel('Relative importance')
        right.pyplot(fig2)

# ── Data Explorer ──────────────────────────────────────────────────────────────
with tab4:
    st.header("📊 Data Explorer")
    st.write("Descriptive charts from the full teff germplasm dataset (2,082 accessions).")

    try:
        eda = load_eda_data()
    except Exception as e:
        st.error(f"Could not load teff.xlsx for EDA: {e}")
        st.stop()

    PALETTE = {'Early': '#2196F3', 'Medium': '#4CAF50', 'Late': '#FF9800'}
    ORDER   = ['Early', 'Medium', 'Late']

    # ── Row 1: class balance + heading date vs maturity ───────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("Class distribution (Maturity)")
        counts = eda['Mdatec'].value_counts().reindex(ORDER)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(counts.index, counts.values,
                      color=[PALETTE[c] for c in counts.index], edgecolor='white', width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Maturity class'); ax.set_ylabel('Count')
        ax.set_title('Number of accessions per maturity class')
        sns.despine(ax=ax)
        st.pyplot(fig)
        st.caption("The dataset is imbalanced — Medium dominates, Early is rare.")

    with r1c2:
        st.subheader("Heading date vs Maturity")
        hd_order = ['Early', 'Medium', 'Late']
        ct = pd.crosstab(eda['Hdatec'], eda['Mdatec']).reindex(hd_order, fill_value=0)[ORDER]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct.plot(kind='bar', ax=ax, color=[PALETTE[c] for c in ORDER],
                edgecolor='white', width=0.6)
        ax.set_xlabel('Heading date category'); ax.set_ylabel('Count')
        ax.set_title('Heading date vs Maturity class')
        ax.legend(title='Maturity', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='x', rotation=0)
        sns.despine(ax=ax)
        st.pyplot(fig)
        st.caption("Heading date and maturity are strongly correlated — earlier heading → earlier maturity.")

    st.divider()

    # ── Row 2: grain filling period + altitude distributions ──────────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Grain filling period by Maturity")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for cls in ORDER:
            vals = eda.loc[eda['Mdatec'] == cls, 'GfilP'].dropna()
            ax.hist(vals, bins=20, alpha=0.6, label=cls, color=PALETTE[cls], edgecolor='white')
        ax.set_xlabel('Grain filling period (days)'); ax.set_ylabel('Count')
        ax.set_title('Distribution of grain filling period')
        ax.legend(title='Maturity')
        sns.despine(ax=ax)
        st.pyplot(fig)
        st.caption("Late-maturing accessions tend to have longer grain filling periods.")

    with r2c2:
        st.subheader("Altitude by Maturity")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        data_by_class = [eda.loc[eda['Mdatec'] == cls, 'Altitude'].dropna() for cls in ORDER]
        import matplotlib
        _mpl_ver = tuple(int(x) for x in matplotlib.__version__.split('.')[:2])
        _bp_label_kw = 'tick_labels' if _mpl_ver >= (3, 9) else 'labels'
        bp = ax.boxplot(data_by_class, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2),
                        **{_bp_label_kw: ORDER})
        for patch, cls in zip(bp['boxes'], ORDER):
            patch.set_facecolor(PALETTE[cls])
            patch.set_alpha(0.7)
        ax.set_xlabel('Maturity class'); ax.set_ylabel('Altitude (m)')
        ax.set_title('Altitude distribution by maturity class')
        sns.despine(ax=ax)
        st.pyplot(fig)
        st.caption("Late-maturing types are typically found at higher altitudes with longer growing seasons.")

    st.divider()

    # ── Row 3: annual precipitation + panicle type heatmap ───────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.subheader("Annual precipitation by Maturity")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for cls in ORDER:
            vals = eda.loc[eda['Mdatec'] == cls, 'bio12_1970_2000'].dropna()
            sns.kdeplot(vals, ax=ax, label=cls, color=PALETTE[cls], fill=True, alpha=0.3)
        ax.set_xlabel('Annual precipitation — bio12 (mm)'); ax.set_ylabel('Density')
        ax.set_title('Precipitation distribution by maturity class')
        ax.legend(title='Maturity')
        sns.despine(ax=ax)
        st.pyplot(fig)
        st.caption("Early-maturing accessions are more common in drier zones.")

    with r3c2:
        st.subheader("Panicle type vs Maturity")
        ptype_order = ['very loose', 'loose', 'semi-loose', 'compact', 'very compact']
        ct2 = pd.crosstab(eda['Ptype'], eda['Mdatec'])
        ct2 = ct2.reindex([p for p in ptype_order if p in ct2.index], fill_value=0)
        ct2 = ct2[[c for c in ORDER if c in ct2.columns]]
        # Normalise to % within each panicle type
        ct2_pct = ct2.div(ct2.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ct2_pct.plot(kind='barh', ax=ax, color=[PALETTE[c] for c in ct2_pct.columns],
                     edgecolor='white', width=0.6)
        ax.set_xlabel('% of accessions'); ax.set_ylabel('Panicle type')
        ax.set_title('Maturity class share by panicle type (%)')
        ax.legend(title='Maturity', bbox_to_anchor=(1, 1))
        sns.despine(ax=ax)
        st.pyplot(fig)
        st.caption("Compact panicle types are more associated with late maturity.")

    st.divider()

    # ── Summary stats table ───────────────────────────────────────────────────
    st.subheader("Summary statistics by Maturity class")
    num_cols = ['GfilP', 'Altitude', 'ph_kcl_60cm', 'bio12_1970_2000', 'bio5_1970_2000']
    summary = eda.groupby('Mdatec')[num_cols].agg(['mean', 'std']).round(2)
    summary.columns = [f"{col} ({stat})" for col, stat in summary.columns]
    st.dataframe(summary.reindex(ORDER), use_container_width=True)
