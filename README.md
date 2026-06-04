#  Ames Housing — Sale Price Prediction with SVR

An end-to-end machine learning pipeline on the Ames Housing dataset, covering EDA, preprocessing, feature engineering, and Support Vector Regression (RBF kernel) with GridSearchCV tuning.

---

##  Overview

> Predict house sale prices using SVR with a full leakage-free preprocessing pipeline.

| What | Details |
|------|---------|
| **Dataset** | Ames Housing (`AmesHousing.csv`) |
| **Target** | `SalePrice` — log-transformed (`log1p`) |
| **Algorithm** | Support Vector Regression · RBF kernel · GridSearchCV |
| **Encoding** | Target encoding for categorical features + ordinal encoding for quality columns |

---

##  Pipeline

| # | Stage | Description |
|---|-------|-------------|
| 1 | **EDA** | Missing values, skewness audit, outlier detection, correlation analysis, sparse features |
| 2 | **Cleaning** | NaN → `'None'` for absent features; median imputation for structural columns; catch-all fallback |
| 3 | **Outlier Removal** | Drop rows with `Gr Liv Area > 4000` and `SalePrice < $300K` (anomalous low-price large homes) |
| 4 | **Feature Engineering** | 9 new features — area totals, age, bath count, porch area, binary indicators |
| 5 | **Ordinal Encoding** | Quality/condition columns mapped to numeric scale (None=0 → Ex=5) |
| 6 | **Log Transform** | `log1p` applied to all features with `|skew| > 0.75` + target `SalePrice` |
| 7 | **Drop Redundant Columns** | Near-zero-variance, sparse, and highly correlated columns removed |
| 8 | **Target Encoding** | Categorical columns encoded using mean `SalePrice` per category |
| 9 | **Scaling** | `StandardScaler` — critical for SVR (RBF kernel uses Euclidean distances) |
| 10 | **SVR + GridSearchCV** | RBF kernel · 5-fold CV · scored on neg-RMSE · `n_jobs=-1` |
| 11 | **Evaluation** | R², RMSE (log + USD), MAE, 5-fold CV R², residual diagnostics |

---

##  Feature Engineering

| Feature | Formula |
|---------|---------|
| `TotalSF` | `Total Bsmt SF + 1st Flr SF + 2nd Flr SF` |
| `HouseAge` | `Yr Sold − Year Built` |
| `RemodAge` | `Yr Sold − Year Remod/Add` |
| `TotalBath` | `Full Bath + Bsmt Full Bath + 0.5 × (Half Bath + Bsmt Half Bath)` |
| `TotalPorchSF` | Sum of all porch/deck areas |
| `HasGarage` | Binary: `Garage Area > 0` |
| `HasPool` | Binary: `Pool Area > 0` |
| `HasFireplace` | Binary: `Fireplaces > 0` |
| `Has2ndFloor` | Binary: `2nd Flr SF > 0` |

---

##  Dropped Columns

| Column | Reason |
|--------|--------|
| `Street` | 99%+ Pave — no variance |
| `Utilities` | Almost entirely `AllPub` |
| `Pool Area` | Replaced by `HasPool` |
| `Garage Cars` | Highly correlated with `Garage Area` (r > 0.88) |
| `Garage Yr Blt` | Redundant with `HouseAge` |
| `Year Built` | Replaced by `HouseAge` |
| `Year Remod/Add` | Replaced by `RemodAge` |
| `Misc Val`, `Low Qual Fin SF`, `3Ssn Porch` | Very sparse |

---

##  SVR Hyperparameter Grid

| Parameter | Values Searched |
|-----------|----------------|
| `C` | 1, 10, 50, 100 |
| `epsilon` | 0.01, 0.05, 0.1 |
| `gamma` | `scale`, `auto` |
| `kernel` | RBF (fixed) |
| `cv` | 5-fold stratified |

---

##  Results

| Metric | Value |
|--------|-------|
| **Test R²** | See notebook output |
| **RMSE (log scale)** | See notebook output |
| **RMSE (USD)** | See notebook output |
| **CV R² (mean ± std)** | See notebook output |

> Run the notebook to get exact scores for your environment and data split.

---

##  Visualisations Generated

| Plot | Description |
|------|-------------|
| Missing values bar | Top 15 columns by missing % |
| SalePrice distribution | Raw vs `log1p` — skewness comparison |
| Skewed features | Bar chart of `|skew|` for top numerical features |
| Outlier scatter | `Gr Liv Area` vs `SalePrice` with flagged anomalies |
| Top correlations + heatmap | Top 12 features correlated with `SalePrice` + multicollinearity |
| Sparse features | % zero values for near-empty columns |
| Q-Q plot | Normality check on log-transformed target |
| Skewness before/after | Grouped bar — improvement from `log1p` transform |
| Post-cleaning scatter | `Gr Liv Area` vs `SalePrice` after outlier removal |
| Engineered features | Scatter of `TotalSF`, `HouseAge`, `TotalBath`, `TotalPorchSF` vs target |
| Actual vs Predicted | USD scale with perfect-fit reference line |
| Residual plot | Residuals vs fitted values |
| Residual distribution | Histogram + normal fit overlay |

---

##  Getting Started

### 1. Clone the repo
```bash
git https://github.com/Hadaba-eddition/ames-housing-svr.git
cd ames-housing-svr
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 3. Set your data path
In the notebook, update:
```python
df = pd.read_csv('AmesHousing.csv')
```

### 4. Run
```bash
jupyter notebook HADABA_AMES_HOUSING.ipynb
```

---

##  Requirements

| Package | Minimum |
|---------|---------|
| Python | 3.8 |
| scikit-learn | 1.0 |
| pandas | 1.3 |
| numpy | 1.21 |
| matplotlib | 3.4 |
| seaborn | 0.11 |
| scipy | 1.7 |

---

##  Repository Structure

```
ames-housing-svr/
├── HADABA_AMES_HOUSING.ipynb           # Main notebook
└── README.md
```

---

##  Citation

Dataset: [Ames Housing Dataset — Dean De Cock (2011)](http://jse.amstat.org/v19n3/decock.pdf)

> De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
