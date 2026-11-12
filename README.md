# AltScore Data Science Challenge (E10) — Geospatial Cost of Living Prediction

End-to-end machine learning pipeline built to estimate the normalized `cost_of_living` index (ranging from `0` to `1`) across H3 resolution-9 hexagonal zones in Latin America, evaluated under **Mean Absolute Error (MAE)**.

---

## Architecture & Methodology

### 1. Scalable Streaming Processing (`PyArrow` + `H3`)
The raw mobility dataset (`mobility_data.parquet`) contains **340.4M+ device-level GPS records**. To process this volume without exceeding system RAM limits, the pipeline reads the Parquet file in streaming batches (`2,000,000` rows per chunk), maps `(lat, lon)` coordinates to **Uber H3 Resolution 9** cells, and aggregates behavioral metrics into a local cache (`hex_mobility_cache.parquet`).

### 2. External Data Enrichment
Per competition guidelines, public external data was integrated to capture topographical and urban context:
- **Source:** [Open-Meteo Elevation API](https://open-meteo.com/en/docs/elevation-api) (`https://api.open-meteo.com/v1/elevation`)
- **Features Extracted:**
  - `elevation_m`: Altitude (meters above sea level) at each hexagon's centroid.
  - `neighbor_avg_elevation_m` & `elevation_diff_neighbors_m`: Local terrain gradient comparing each cell against its immediate `k=1` H3 ring neighbors.
  - `elev_diff_from_city_mean_m`: Relative altitude compared to the macro-region (H3 resolution 4) mean, capturing socioeconomic elevation patterns common in LATAM cities.

### 3. Feature Engineering & Spatial Interpolation
- **Mobility & Temporal Behavior:** Total pings, unique devices, pings per device, weekend vs. weekday ratios, nighttime activity (`22:00–06:00`), and business-hour density.
- **Urban Activity Context:** Distance (`km`) from each hexagon to the weighted activity centroid of its parent H3 resolution-4 region (`dist_to_activity_center_km`).
- **Leak-Free Multi-Scale Spatial IDW:** Inverse Distance Weighting (`idw_cost_k3`, `idw_cost_k7`, `idw_cost_k15`) computed strictly inside each `GroupKFold` split (grouped by H3 resolution-6 parent cells) to prevent spatial data leakage.

### 4. Modeling & Validation
- **Validation Strategy:** 5-Fold `GroupKFold` grouped by parent H3 resolution-6 cells (`hex_res6`).
- **Model Ensemble (L1 / MAE Objective):**
  - **LightGBM Regressor** (`objective="regression_l1"`, weight `0.45`)
  - **ExtraTrees Regressor** (`criterion="absolute_error"`, weight `0.35`)
  - **Gradient Boosting Regressor** (`loss="absolute_error"`, weight `0.20`)

---

## Results

| Model / Stage | Out-of-Fold (OOF) MAE |
| :--- | :---: |
| LightGBM (L1) | `0.11322` |
| ExtraTrees (L1) | `0.11466` |
| Gradient Boosting (L1) | `0.11780` |
| **Weighted Ensemble (Final)** | **`0.11319`** |

- **Kaggle Public Leaderboard MAE:** `0.15283`
- **Kaggle Private Leaderboard MAE:** `0.16250`

---

## Project Structure

```text
├── reto10.py              # Main pipeline: streaming aggregation, API enrichment, IDW & ensemble training
├── requirements.txt       # Python dependencies
├── README.md              # Methodology, external sources, and reproducibility guide
└── .gitignore             # Excludes raw Parquet/CSV datasets and local cache files