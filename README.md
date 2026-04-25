# CarbonSense India

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
``
