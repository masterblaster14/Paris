# CarbonSense India v6.0 — Flask + ML Backend

## 4 ML Models
| Model | Features | Notes |
|---|---|---|
| **Hybrid Ensemble** | Poly-3 + Random Forest + GBR (optimised weights) | Best R² |
| **GDP + Population** | gdp, population | Economic indicator model |
| **Gradient Boosting** | year, coal, oil, gas, cement, population | 500 trees, sequential |
| **SVR (RBF)** | year, coal, oil, gas, cement, population | Kernel-based, good on small data |

All models trained on **OWID India CO₂ dataset, post-1990** data.

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure owid-co2-data_india.xlsx is in this folder
# (already included)

# 3. Start Flask
python app.py

# 4. Open browser
open http://localhost:5000
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/models/summary` | R², MAE, RMSE, CV scores for all 4 models |
| `GET /api/models/projections` | 2025–2040 forecasts from all models |
| `GET /api/models/test_comparison` | Test set actual vs predicted |
| `GET /api/models/feature_importance` | RF & GBR feature importances |
| `GET /api/models/residuals` | Residual arrays for all models |
| `POST /api/predict` | Custom prediction with user-provided inputs |

### POST /api/predict example
```json
{
  "year": 2030,
  "coal_co2": 2200,
  "oil_co2": 800,
  "gas_co2": 160,
  "cement_co2": 200,
  "population": 1450000000
}
```

## New Frontend Tab: 🤖 ML Compare
- **Model cards** showing R², MAE, RMSE for each model
- **Projection chart** — all 4 models 2025–2040 overlaid
- **Test comparison** — actual vs predicted on held-out test years
- **Residual distributions** — histogram per model
- **Feature importance** — RF vs GBR bar charts
- **Radar chart** — normalised comparison across all metrics
- **Cross-validation** — 5-fold CV R² ± std
- **Custom prediction** — enter your own inputs, get all 4 model outputs
