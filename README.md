# MatRisk AI

[![Open in Streamlit]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg](https://matrisk-ai-eq3993fwi9fc3pc88uenwh.streamlit.app/))](https://matrisk-ai.streamlit.app)

> *Atoms → Physics → Finance*

Material degradation is invisible until it isn't. MatRisk AI makes it legible — translating atomic-level chemistry into survival probabilities, and survival probabilities into dollar-denominated risk.

---

## What It Does

Five lenses on a single problem — the lifecycle of physical infrastructure:

| Module | Input | Output |
|---|---|---|
| **Material Lab** | Chemical formula | Bulk/Shear moduli, Poisson's Ratio (PINN-enforced) |
| **Risk Command Center** | Bridge condition ratings | EL, EAD, PD via DeepSurv |
| **Historical Validation** | Failure records (DS6) | Backtested survival accuracy |
| **ESG Calculator** | Virgin vs. recycled specs | Carbon savings without structural compromise |
| **Market Signals** | MQI + commodity futures | Trading alpha from material quality trends |

---

## The Core Idea

```
Chemical Formula
      ↓
  PINN Model  ──→  Thermodynamically-valid material properties
      ↓
DeepSurv Engine ──→  Failure probability over time (PD)
      ↓
Credit Risk Layer ──→  EL = PD × LGD × EAD
      ↓
ESG + Market Overlay ──→  Allocation decisions
```

Most infrastructure risk models skip the first two steps entirely. This one doesn't.

---

## Stack

```
Frontend       Streamlit · Plotly
Data           Pandas · NumPy
ML             Scikit-Learn · XGBoost · Optuna
Deep Learning  PyTorch · PyCox (DeepSurv)
Materials      Matminer
```

---

## Datasets

Six purpose-built datasets ship with the project:

- `DS1` — 5,500 material property records (atomic level)
- `DS2` — 10 years of commodity price time-series
- `DS3` — 5,000 bridge infrastructure condition records
- `DS4` — Pre-computed MQI and market trend features (daily)
- `DS5` — Element price history (monthly)
- `DS6` — 2,000 real-world structural failure events

---

## Quickstart

```bash
git clone https://github.com/your-username/MatRisk-AI.git
cd MatRisk-AI

pip install -r requirements.txt

python -m streamlit run app.py
# → http://localhost:8501
```

Windows users: `py -m streamlit run app.py` or `streamlit run app.py`

---

## Project Layout

```
MatRisk-Data-Set-Sample/
├── app.py                          # 5-page Streamlit dashboard
├── dataset_integration.py          # Core modelling pipeline
├── requirements.txt
├── DS1_material_properties_5500.csv
├── DS2_commodity_prices_10yr.csv
├── DS3_infrastructure_bridges_5000.csv
├── DS4_crossdomain_features_daily.csv
├── DS5_element_prices_monthly.csv
└── DS6_historical_failures_2000.csv
```

*Built at the intersection of materials science, survival analysis, and financial risk.*
