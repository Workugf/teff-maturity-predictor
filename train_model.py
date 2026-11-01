"""
train_model.py
--------------
Replicates the training pipeline from the notebook and saves:
  - teff_pipeline.joblib
  - teff_insights.joblib
Run once before launching streamlit_app.py.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

# ── Constants (identical to notebook) ─────────────────────────────────────────
RANDOM_STATE = 42
TARGET_MAP   = {'Early': 0, 'Medium': 1, 'Late': 2}
CLASS_NAMES  = ['Early', 'Medium', 'Late']
HDATEC_ORDER = ['Early', 'Medium', 'Late']
PTYPE_ORDER  = ['very loose', 'loose', 'semi-loose', 'compact', 'very compact']
NUM_FEATURES = ['GfilP', 'Altitude', 'ph_kcl_60cm', 'bio12_1970_2000',
                'Thermal_Moisture_Index', 'Growth_Season_Moisture_Index']
CAT_FEATURES = ['Hdatec', 'Ptype']
FEATURES     = NUM_FEATURES + CAT_FEATURES
CAT_IDX      = list(range(len(NUM_FEATURES), len(FEATURES)))

curated_features = [
    'Hdatec', 'Gfpc', 'GfilP', 'Ptype', 'SC', 'PCHdate', 'SBP', 'PCMdate',
    'Altitude', 'Plc', 'Edatec', 'ph_kcl_60cm', 'ph_ho_60cm',
    'bio12_1970_2000', 'bio13_1970_2000', 'bio16_1970_2000',
    'bio19_1970_2000', 'bio5_1970_2000', 'bio8_1970_2000', 'Mdatec'
]

# ── Feature engineering ───────────────────────────────────────────────────────
def add_engineered_features(frame):
    out = frame.copy()
    out['Thermal_Moisture_Index']      = out['bio5_1970_2000']  / (out['bio12_1970_2000'] + 1)
    out['Growth_Season_Moisture_Index'] = out['bio16_1970_2000'] / (out['bio12_1970_2000'] + 1)
    return out

# ── Pipeline factory ──────────────────────────────────────────────────────────
def make_preprocessor():
    return ColumnTransformer([
        ('num', SkPipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler())
        ]), NUM_FEATURES),
        ('cat', OrdinalEncoder(
            categories=[HDATEC_ORDER, PTYPE_ORDER],
            handle_unknown='use_encoded_value', unknown_value=-1
        ), CAT_FEATURES),
    ])

def make_pipeline(estimator):
    return ImbPipeline([
        ('prep',  make_preprocessor()),
        ('smote', SMOTENC(categorical_features=CAT_IDX, random_state=RANDOM_STATE)),
        ('model', estimator),
    ])

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading teff.xlsx ...")
data = pd.read_excel('teff.xlsx')
print(f"  Loaded {data.shape[0]} rows, {data.shape[1]} columns")

# ── 2. Feature selection ──────────────────────────────────────────────────────
existing_cols = [c for c in curated_features if c in data.columns]
df = data[existing_cols].copy()

# ── 3. Cleaning & mapping ─────────────────────────────────────────────────────
cols_to_check = ['Hdatec', 'Gfpc', 'GfilP', 'Ptype', 'SC', 'PCHdate', 'SBP', 'PCMdate', 'Plc', 'Edatec']
for col in cols_to_check:
    if col in df.columns:
        df = df[df[col] != 86]
        df = df[df[col] != '86']

ptype_map   = {1:'very loose',2:'loose',3:'semi-loose',4:'compact',5:'very compact',
               '1':'very loose','2':'loose','3':'semi-loose','4':'compact','5':'very compact'}
hdatec_map  = {1:'Early',2:'Medium',3:'Late','1':'Early','2':'Medium','3':'Late'}
mdatec_map  = {1:'Early',2:'Medium',3:'Late','1':'Early','2':'Medium','3':'Late'}
edatec_map  = {1:'Early',2:'Medium',3:'Late','1':'Early','2':'Medium','3':'Late'}

if 'Ptype'   in df.columns: df['Ptype']   = df['Ptype'].astype(str).str.strip().map(ptype_map)
if 'Hdatec'  in df.columns: df['Hdatec']  = df['Hdatec'].map(hdatec_map).fillna(df['Hdatec'])
if 'Mdatec'  in df.columns: df['Mdatec']  = df['Mdatec'].map(mdatec_map).fillna(df['Mdatec'])
if 'Edatec'  in df.columns: df['Edatec']  = df['Edatec'].map(edatec_map).fillna(df['Edatec'])

df = df.dropna(subset=['Ptype'])

# Drop Gfpc (binned copy of GfilP)
if 'Gfpc' in df.columns:
    df = df.drop(columns=['Gfpc'])

# Drop duplicate rows
df = df.drop_duplicates()
print(f"  After cleaning: {df.shape[0]} rows")

# ── 4. Train / test split ─────────────────────────────────────────────────────
X = df.drop(columns=['Mdatec'])
y = df['Mdatec']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

y_train_enc = y_train.map(TARGET_MAP).astype(int)
y_test_enc  = y_test.map(TARGET_MAP).astype(int)

assert y_train_enc.notna().all(), "Unmapped labels in y_train!"
assert y_test_enc.notna().all(),  "Unmapped labels in y_test!"

X_train_fe = add_engineered_features(X_train)[FEATURES]
X_test_fe  = add_engineered_features(X_test)[FEATURES]
print(f"  Train: {X_train_fe.shape}  Test: {X_test_fe.shape}")

# ── 5. Hyperparameter tuning ──────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grid = {
    'model__n_estimators':      [50, 100, 200],
    'model__max_depth':         [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
}
print("Running GridSearchCV (this may take a minute) ...")
rf_grid = GridSearchCV(
    make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
    param_grid=param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, refit=True
)
rf_grid.fit(X_train_fe, y_train_enc)
print(f"  Best params : {rf_grid.best_params_}")
print(f"  Best CV F1  : {rf_grid.best_score_:.4f}")

# ── 6. Test-set evaluation ────────────────────────────────────────────────────
best   = rf_grid.best_estimator_
y_pred = best.predict(X_test_fe)

print("\n=== Test-set report ===")
print(classification_report(y_test_enc, y_pred, target_names=CLASS_NAMES, digits=3))

cm = confusion_matrix(y_test_enc, y_pred)
test_accuracy = float(accuracy_score(y_test_enc, y_pred))
test_macro_f1 = float(f1_score(y_test_enc, y_pred, average='macro'))
print(f"Accuracy: {test_accuracy:.4f}  Macro-F1: {test_macro_f1:.4f}")

# ── 7. Feature importance ─────────────────────────────────────────────────────
imp_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': best.named_steps['model'].feature_importances_
}).sort_values('Importance', ascending=False)

perm = permutation_importance(
    best, X_test_fe, y_test_enc,
    scoring='f1_macro', n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
)
perm_df = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': perm.importances_mean,
    'Std':        perm.importances_std
}).sort_values('Importance', ascending=False)

# ── 8. Save artifacts ─────────────────────────────────────────────────────────
inference_model = SkPipeline([
    ('prep',  best.named_steps['prep']),
    ('model', best.named_steps['model'])
])
assert np.array_equal(inference_model.predict(X_test_fe), best.predict(X_test_fe))

joblib.dump(inference_model, 'teff_pipeline.joblib')

insights = {
    'class_names':             CLASS_NAMES,
    'confusion_matrix':        cm,
    'feature_importance':      imp_df.reset_index(drop=True),
    'permutation_importance':  perm_df.reset_index(drop=True),
    'metrics': {
        'test_accuracy': test_accuracy,
        'test_macro_f1': test_macro_f1,
        'cv_macro_f1':   float(rf_grid.best_score_),
    },
}
joblib.dump(insights, 'teff_insights.joblib')

print("\nSaved: teff_pipeline.joblib  teff_insights.joblib")
print("Ready — run:  streamlit run streamlit_app.py")
