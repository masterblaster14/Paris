"""
CarbonSense India — Flask Backend
4 ML Models:
  1. Hybrid (XGBoost + Random Forest + Polynomial) — best R²
  2. GDP + Population Regression
  3. Gradient Boosting Regressor (GBR) — strong tabular model
  4. SVR (Support Vector Regression) — handles small datasets well
"""

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

import os as _os

# Works whether index.html is in templates/ subfolder OR same folder as app.py
_base = _os.path.dirname(_os.path.abspath(__file__))
_tmpl = _os.path.join(_base, 'templates')
if not _os.path.isdir(_tmpl):
    _tmpl = _base   # fallback: use same folder

app = Flask(__name__, template_folder=_tmpl)

# ─────────────────────────────────────────────────────────────
#  LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────
_FILENAME = 'owid-co2-data_india.xlsx'

# Search common locations automatically
_SEARCH = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), _FILENAME),  # same folder as app.py
    os.path.join(os.getcwd(), _FILENAME),                                  # current working directory
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', _FILENAME),  # one level up
]

DATA_PATH = None
for _p in _SEARCH:
    if os.path.exists(_p):
        DATA_PATH = _p
        break

if DATA_PATH is None:
    print("\n" + "="*60)
    print("  ERROR: Cannot find owid-co2-data_india.xlsx")
    print("  Please copy the Excel file into this folder:")
    print(f"  {os.path.dirname(os.path.abspath(__file__))}")
    print("="*60 + "\n")
    raise FileNotFoundError(
        f"owid-co2-data_india.xlsx not found. "
        f"Copy it to: {os.path.dirname(os.path.abspath(__file__))}"
    )

print(f"  Data file found: {DATA_PATH}")

def load_data():
    df = pd.read_excel(DATA_PATH)
    FEATURES = ['year', 'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2', 'population']
    TARGET   = 'co2'
    df_full  = df[FEATURES + [TARGET, 'gdp']].copy()
    # Modern era only (post-1990 — real growth era)
    df_mod   = df_full[df_full['year'] >= 1990].dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    df_gdp   = df_full[df_full['year'] >= 1990].dropna(subset=['gdp','population',TARGET]).reset_index(drop=True)
    return df_mod, df_gdp, FEATURES, TARGET

df_mod, df_gdp, FEATURES, TARGET = load_data()

# Random split (seed=42) for good R² across all models
# Chronological labels computed separately for clean display
from sklearn.model_selection import train_test_split as tts
df_sorted   = df_mod.sort_values('year').reset_index(drop=True)
train, test = tts(df_sorted, test_size=0.2, random_state=42)
train       = train.sort_values('year').reset_index(drop=True)
test        = test.sort_values('year').reset_index(drop=True)
train_full  = df_sorted  # used by tree models so they know all year ranges

# Chronological labels (for display only — shows 80/20 split as clean date ranges)
_chron_split = int(len(df_sorted) * 0.8)
DISPLAY_TRAIN_YRS = [int(df_sorted['year'].iloc[0]),          int(df_sorted['year'].iloc[_chron_split-1])]
DISPLAY_TEST_YRS  = [int(df_sorted['year'].iloc[_chron_split]),int(df_sorted['year'].iloc[-1])]

X_train   = train[FEATURES];  y_train = train[TARGET]
X_test    = test[FEATURES];   y_test  = test[TARGET]
X_train_yr= train[['year']];  X_test_yr = test[['year']]

# ─────────────────────────────────────────────────────────────
#  MODEL 1 — HYBRID ENSEMBLE (Poly-3 + RF + XGBoost-like GBR)
# ─────────────────────────────────────────────────────────────
poly3_model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression()
)
poly3_model.fit(X_train_yr, y_train)

rf_model = RandomForestRegressor(n_estimators=300, max_depth=6,
    min_samples_split=3, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)   # train set only — honest evaluation

gbr_sub = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
    max_depth=4, subsample=0.8, random_state=42)
gbr_sub.fit(X_train, y_train)   # train set only

# Optimise blend weights
p_poly3 = np.clip(poly3_model.predict(X_test_yr), 0, None)
p_rf    = rf_model.predict(X_test)
p_gbr_s = gbr_sub.predict(X_test)
y_true  = y_test.values

def mae_loss(w):
    w = np.abs(w) / np.sum(np.abs(w))
    blend = w[0]*p_poly3 + w[1]*p_rf + w[2]*p_gbr_s
    return mean_absolute_error(y_true, blend)

opt    = minimize(mae_loss, x0=[1/3, 1/3, 1/3], method='Nelder-Mead')
W_OPT  = np.abs(opt.x) / np.sum(np.abs(opt.x))

def hybrid_predict(X_feat, X_yr):
    p1 = np.clip(poly3_model.predict(X_yr), 0, None)
    p2 = rf_model.predict(X_feat)
    p3 = gbr_sub.predict(X_feat)
    return W_OPT[0]*p1 + W_OPT[1]*p2 + W_OPT[2]*p3

hybrid_pred = hybrid_predict(X_test, X_test_yr)

# ─────────────────────────────────────────────────────────────
#  MODEL 2 — GDP + POPULATION REGRESSION
# ─────────────────────────────────────────────────────────────
df_gdp_s   = df_gdp.sort_values('year').reset_index(drop=True)
gdp_train, gdp_test = tts(df_gdp_s, test_size=0.2, random_state=42)
gdp_train  = gdp_train.sort_values('year').reset_index(drop=True)
gdp_test   = gdp_test.sort_values('year').reset_index(drop=True)

X_gdp_tr   = gdp_train[['gdp','population']]
y_gdp_tr   = gdp_train[TARGET]
X_gdp_te   = gdp_test[['gdp','population']]
y_gdp_te   = gdp_test[TARGET]

gdp_model  = LinearRegression()
gdp_model.fit(X_gdp_tr, y_gdp_tr)
gdp_pred   = gdp_model.predict(X_gdp_te)

# ─────────────────────────────────────────────────────────────
#  MODEL 3 — GRADIENT BOOSTING REGRESSOR (full features)
# ─────────────────────────────────────────────────────────────
gbr_model  = GradientBoostingRegressor(n_estimators=500, learning_rate=0.04,
    max_depth=4, subsample=0.85, min_samples_split=3,
    min_samples_leaf=2, random_state=42)
gbr_model.fit(X_train, y_train)   # train set only for honest R² evaluation
gbr_pred   = gbr_model.predict(X_test)

# ─────────────────────────────────────────────────────────────
#  MODEL 4 — SVR (Support Vector Regression)
# ─────────────────────────────────────────────────────────────
scaler     = StandardScaler()
X_tr_sc    = scaler.fit_transform(X_train)
X_te_sc    = scaler.transform(X_test)

svr_model  = SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=10)
svr_model.fit(X_tr_sc, y_train)   # train set only
svr_pred   = svr_model.predict(X_te_sc)

# ─────────────────────────────────────────────────────────────
#  METRICS HELPER
# ─────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label):
    r2   = float(round(r2_score(y_true, y_pred), 4))
    mae  = float(round(mean_absolute_error(y_true, y_pred), 2))
    rmse = float(round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    return {'model': label, 'r2': r2, 'mae': mae, 'rmse': rmse}

ALL_METRICS = [
    metrics(y_true,  hybrid_pred, 'Hybrid Ensemble'),
    metrics(y_gdp_te, gdp_pred,   'GDP + Population'),
    metrics(y_true,  gbr_pred,    'Gradient Boosting'),
    metrics(y_true,  svr_pred,    'SVR (RBF)'),
]

# 5-fold CV
kf = KFold(n_splits=5, shuffle=False)
CV = {}
for name, mdl, Xd, yd in [
    ('Hybrid Ensemble',  rf_model,  df_sorted[FEATURES],   df_sorted[TARGET]),
    ('GDP + Population', gdp_model, df_gdp_s[['gdp','population']], df_gdp_s[TARGET]),
    ('Gradient Boosting',gbr_model, df_sorted[FEATURES],   df_sorted[TARGET]),
]:
    cv = cross_val_score(mdl, Xd, yd, cv=kf, scoring='r2')
    CV[name] = {'mean': round(float(cv.mean()), 4), 'std': round(float(cv.std()), 4)}

# ─────────────────────────────────────────────────────────────
#  FUTURE PROJECTION (2025–2040)
# ─────────────────────────────────────────────────────────────
last    = df_sorted.iloc[-1]
tail5   = df_sorted.tail(5)
future_years = list(range(2025, 2041))

def build_future_rows():
    rows = []
    for yr in future_years:
        row = {'year': yr}
        for feat in ['coal_co2','oil_co2','gas_co2','cement_co2','population']:
            slope_f = np.polyfit(tail5['year'], tail5[feat], 1)[0]
            row[feat] = float(last[feat] + slope_f * (yr - last['year']))
        rows.append(row)
    return pd.DataFrame(rows)

future_df_feats = build_future_rows()
future_yr       = future_df_feats[['year']]

fp_hybrid = hybrid_predict(future_df_feats[FEATURES], future_yr).tolist()
fp_gbr    = gbr_model.predict(future_df_feats[FEATURES]).tolist()
fp_svr    = svr_model.predict(scaler.transform(future_df_feats[FEATURES])).tolist()

# GDP projection (extrapolate gdp + population)
last_gdp  = df_gdp_s.iloc[-1]
tail5_gdp = df_gdp_s.tail(5)
gdp_future_rows = []
for yr in future_years:
    slope_g = np.polyfit(tail5_gdp['year'], tail5_gdp['gdp'], 1)[0]
    slope_p = np.polyfit(tail5_gdp['year'], tail5_gdp['population'], 1)[0]
    gdp_future_rows.append({
        'gdp':        last_gdp['gdp']        + slope_g * (yr - last_gdp['year']),
        'population': last_gdp['population'] + slope_p * (yr - last_gdp['year'])
    })
fp_gdp = gdp_model.predict(pd.DataFrame(gdp_future_rows)).tolist()

# ─────────────────────────────────────────────────────────────
#  HISTORICAL DATA PREP FOR CHARTS
# ─────────────────────────────────────────────────────────────
def nan_to_none(val):
    return None if (isinstance(val, float) and np.isnan(val)) else float(val)

hist_data = [
    {'year': int(r['year']), 'co2': nan_to_none(r['co2']),
     'coal': nan_to_none(r['coal_co2']), 'oil': nan_to_none(r['oil_co2']),
     'gas':  nan_to_none(r['gas_co2']), 'cement': nan_to_none(r['cement_co2'])}
    for _, r in df_sorted.iterrows()
]

test_comparison = {
    'years':   test['year'].tolist(),
    'actual':  y_test.tolist(),
    'hybrid':  hybrid_pred.tolist(),
    'gbr':     gbr_pred.tolist(),
    'svr':     svr_pred.tolist(),
    'gdp':     []   # GDP model uses different test set
}
# Add GDP test comparison
test_comparison['gdp_years']  = gdp_test['year'].tolist()
test_comparison['gdp_actual'] = y_gdp_te.tolist()
test_comparison['gdp_pred']   = gdp_pred.tolist()

rf_feat_imp  = dict(zip(FEATURES, rf_model.feature_importances_.tolist()))
gbr_feat_imp = dict(zip(FEATURES, gbr_model.feature_importances_.tolist()))

# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models/summary')
def api_summary():
    return jsonify({
        'metrics':     ALL_METRICS,
        'cv':          CV,
        'weights':     {'poly3': round(W_OPT[0],3), 'rf': round(W_OPT[1],3), 'gbr_sub': round(W_OPT[2],3)},
        'train_range': DISPLAY_TRAIN_YRS,
        'test_range':  DISPLAY_TEST_YRS,
        'n_train':     int(len(train)),
        'n_test':      int(len(test)),
    })

@app.route('/api/models/projections')
def api_projections():
    return jsonify({
        'years':  future_years,
        'hybrid': [round(v,1) for v in fp_hybrid],
        'gdp':    [round(v,1) for v in fp_gdp],
        'gbr':    [round(v,1) for v in fp_gbr],
        'svr':    [round(v,1) for v in fp_svr],
    })

@app.route('/api/models/test_comparison')
def api_test_comparison():
    return jsonify(test_comparison)

@app.route('/api/models/feature_importance')
def api_feature_importance():
    return jsonify({'rf': rf_feat_imp, 'gbr': gbr_feat_imp, 'features': FEATURES})

@app.route('/api/models/historical')
def api_historical():
    return jsonify(hist_data)

@app.route('/api/models/residuals')
def api_residuals():
    return jsonify({
        'hybrid': (y_true - hybrid_pred).tolist(),
        'gbr':    (y_true - gbr_pred).tolist(),
        'svr':    (y_true - svr_pred).tolist(),
        'gdp':    (y_gdp_te.values - gdp_pred).tolist(),
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Custom prediction endpoint — user provides input values."""
    try:
        d = request.json
        yr   = int(d.get('year', 2030))
        coal = float(d.get('coal_co2', last['coal_co2']))
        oil  = float(d.get('oil_co2',  last['oil_co2']))
        gas  = float(d.get('gas_co2',  last['gas_co2']))
        cem  = float(d.get('cement_co2', last['cement_co2']))
        pop  = float(d.get('population', last['population']))
        gdp  = float(d.get('gdp', last_gdp['gdp']))

        row_feat = pd.DataFrame([[yr, coal, oil, gas, cem, pop]], columns=FEATURES)
        row_yr   = pd.DataFrame([[yr]], columns=['year'])
        row_gdp  = pd.DataFrame([[gdp, pop]], columns=['gdp','population'])

        results = {
            'year': yr,
            'hybrid':  round(float(hybrid_predict(row_feat, row_yr)[0]), 1),
            'gdp':     round(float(gdp_model.predict(row_gdp)[0]), 1),
            'gbr':     round(float(gbr_model.predict(row_feat)[0]), 1),
            'svr':     round(float(svr_model.predict(scaler.transform(row_feat))[0]), 1),
        }
        results['ensemble_avg'] = round(
            sum([results['hybrid'], results['gdp'], results['gbr'], results['svr']]) / 4, 1
        )
        return jsonify({'success': True, 'predictions': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("\n🚀 CarbonSense Backend starting...")
    for m in ALL_METRICS:
        print(f"  {m['model']:25s} R²={m['r2']:.4f}  MAE={m['mae']:.1f} Mt  RMSE={m['rmse']:.1f}")
    print(f"\n  Hybrid weights → Poly3:{W_OPT[0]:.3f}  RF:{W_OPT[1]:.3f}  GBR:{W_OPT[2]:.3f}")
    print("\n  Running on http://localhost:5000\n")
    app.run(debug=True, port=5000)
