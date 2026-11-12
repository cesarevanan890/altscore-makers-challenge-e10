import os
import time
import h3
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

# ==========================================
# 1. CARGA DE DATOS Y CACHÉ DE MOVILIDAD
# ==========================================
print("1. Cargando train, test y caché de movilidad...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df_feat = pd.read_parquet("hex_mobility_cache.parquet")

if hasattr(h3, "latlng_to_cell"):
  get_parent = lambda h, r: h3.cell_to_parent(h, r)
  get_center = lambda h: h3.cell_to_latlng(h)
  get_neighbors = lambda h: h3.grid_disk(h, 1)
else:
  get_parent = lambda h, r: h3.h3_to_parent(h, r)
  get_center = lambda h: h3.h3_to_geo(h)
  get_neighbors = lambda h: h3.k_ring(h, 1)

coords = [get_center(h) for h in df_feat["hex_id"]]
df_feat["center_lat"] = [c[0] for c in coords]
df_feat["center_lon"] = [c[1] for c in coords]
df_feat["hex_res8"] = [get_parent(h, 8) for h in df_feat["hex_id"]]
df_feat["hex_res7"] = [get_parent(h, 7) for h in df_feat["hex_id"]]
df_feat["hex_res6"] = [get_parent(h, 6) for h in df_feat["hex_id"]]
df_feat["region_res4"] = [get_parent(h, 4) for h in df_feat["hex_id"]]

# ==========================================
# 2. FUENTE EXTERNA: ELEVACIÓN TOPOGRÁFICA (OPEN-METEO API)
# ==========================================
CACHE_ELEV = "elevation_cache.csv"


def get_elevations(df_hex):
  if os.path.exists(CACHE_ELEV):
    print(f"2. Cargando elevaciones desde caché ({CACHE_ELEV})...")
    return pd.read_csv(CACHE_ELEV)

  print(
      "2. Descargando altitud topográfica (m.s.n.m.) desde Open-Meteo"
      " Elevation API..."
  )
  lats = df_hex["center_lat"].tolist()
  lons = df_hex["center_lon"].tolist()
  elevations = []

  batch_size = 100
  for i in range(0, len(lats), batch_size):
    b_lat = ",".join(f"{x:.5f}" for x in lats[i : i + batch_size])
    b_lon = ",".join(f"{x:.5f}" for x in lons[i : i + batch_size])
    url = f"https://api.open-meteo.com/v1/elevation?latitude={b_lat}&longitude={b_lon}"
    try:
      r = requests.get(url, timeout=15)
      if r.status_code == 200:
        elevations.extend(
            r.json().get("elevation", [np.nan] * len(lats[i : i + batch_size]))
        )
      else:
        elevations.extend([np.nan] * len(lats[i : i + batch_size]))
    except Exception:
      elevations.extend([np.nan] * len(lats[i : i + batch_size]))
    time.sleep(0.15)

  df_e = pd.DataFrame({"hex_id": df_hex["hex_id"], "elevation_m": elevations})
  df_e.to_csv(CACHE_ELEV, index=False)
  print(f"   -> {len(df_e)} altitudes guardadas en {CACHE_ELEV}")
  return df_e


df_elev = get_elevations(df_feat)
df_feat = df_feat.merge(df_elev, on="hex_id", how="left")

# ==========================================
# 3. FEATURES DE VECINDARIO TOPOGRÁFICO Y ACTIVIDAD
# ==========================================
elev_map = df_feat.set_index("hex_id")["elevation_m"].to_dict()
ping_map = df_feat.set_index("hex_id")["total_pings"].to_dict()
dev_map = df_feat.set_index("hex_id")["unique_devices"].to_dict()

nbr_elev_diff, nbr_elev_mean, nbr_pings, nbr_devs = [], [], [], []
for h, el in zip(df_feat["hex_id"], df_feat["elevation_m"]):
  nbrs = [n for n in get_neighbors(h) if n != h]
  nbr_els = [
      elev_map.get(n, el)
      for n in nbrs
      if not np.isnan(elev_map.get(n, np.nan))
  ]
  mean_el = np.mean(nbr_els) if nbr_els else el
  nbr_elev_mean.append(mean_el)
  # Diferencia de altura respecto a sus vecinos (detecta lomas/cerros vs valles)
  nbr_elev_diff.append(el - mean_el if not np.isnan(el) else 0.0)
  nbr_pings.append(np.mean([ping_map.get(n, 0) for n in nbrs]))
  nbr_devs.append(np.mean([dev_map.get(n, 0) for n in nbrs]))

df_feat["neighbor_avg_elevation_m"] = nbr_elev_mean
df_feat["elevation_diff_neighbors_m"] = nbr_elev_diff
df_feat["neighbor_avg_pings"] = nbr_pings
df_feat["neighbor_avg_devices"] = nbr_devs

# Altitud relativa respecto al promedio de toda su ciudad/región (resolución 4)
region_elev = (
    df_feat.groupby("region_res4")["elevation_m"]
    .agg(region_mean_elev="mean", region_std_elev="std")
    .reset_index()
)
df_feat = df_feat.merge(region_elev, on="region_res4", how="left")
df_feat["elev_diff_from_city_mean_m"] = (
    df_feat["elevation_m"] - df_feat["region_mean_elev"]
)

# Distancia al centroide de actividad de cada región (resolución 4)
city_centers = (
    df_feat.groupby("region_res4")
    .apply(
        lambda g: pd.Series({
            "city_lat": np.average(
                g["center_lat"], weights=g["total_pings"] + 1
            ),
            "city_lon": np.average(
                g["center_lon"], weights=g["total_pings"] + 1
            ),
        }),
        include_groups=False,
    )
    .reset_index()
)
df_feat = df_feat.merge(city_centers, on="region_res4", how="left")
df_feat["dist_to_activity_center_km"] = np.sqrt(
    ((df_feat["center_lat"] - df_feat["city_lat"]) * 111.32) ** 2
    + (
        (df_feat["center_lon"] - df_feat["city_lon"])
        * 111.32
        * np.cos(np.radians(df_feat["center_lat"]))
    )
    ** 2
)

# ==========================================
# 4. INTERPOLACIÓN ESPACIAL MULTI-ESCALA (IDW SIN LEAKAGE)
# ==========================================
train_df = train.merge(df_feat, on="hex_id", how="left")
test_df = test[["hex_id"]].merge(df_feat, on="hex_id", how="left")


def add_multiscale_idw(tr_df, target_df):
  tr_coords = np.column_stack([
      tr_df["center_lat"].values * 111.32,
      tr_df["center_lon"].values
      * 111.32
      * np.cos(np.radians(tr_df["center_lat"].values)),
  ])
  tg_coords = np.column_stack([
      target_df["center_lat"].values * 111.32,
      target_df["center_lon"].values
      * 111.32
      * np.cos(np.radians(target_df["center_lat"].values)),
  ])

  tree = cKDTree(tr_coords)
  is_same = len(tr_df) == len(target_df) and np.allclose(tr_coords, tg_coords)
  max_k = 16 if is_same else 15

  dists, idxs = tree.query(tg_coords, k=max_k)
  if is_same:
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

  y_vals = tr_df["cost_of_living"].values[idxs]
  out = {}
  for k in [3, 7, 15]:
    w = 1.0 / (dists[:, :k] + 0.05)
    out[f"idw_cost_k{k}"] = np.sum(w * y_vals[:, :k], axis=1) / np.sum(w, axis=1)
  out["dist_nearest_train_hex_km"] = dists[:, 0]
  return out


test_idw_dict = add_multiscale_idw(train_df, test_df)
for col_name, vals in test_idw_dict.items():
  test_df[col_name] = vals

for col_name in test_idw_dict.keys():
  train_df[col_name] = np.nan

for trn_idx, val_idx in GroupKFold(n_splits=5).split(
    train_df, train_df["cost_of_living"], train_df["hex_res6"]
):
  val_dict = add_multiscale_idw(train_df.iloc[trn_idx], train_df.iloc[val_idx])
  for col_name, vals in val_dict.items():
    train_df.loc[train_df.index[val_idx], col_name] = vals

# ==========================================
# 5. ENSAMBLE DE MODELOS (LIGHTGBM + EXTRATREES + GBR)
# ==========================================
ignore_cols = [
    "hex_id",
    "cost_of_living",
    "hex_res8",
    "hex_res7",
    "hex_res6",
    "region_res4",
    "city_lat",
    "city_lon",
]
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

X = train_df[feature_cols].fillna(0)
y = train_df["cost_of_living"]
groups = train_df["hex_res6"]
X_test = test_df[feature_cols].fillna(0)

print(
    f"\n3. Entrenando Ensamble (LightGBM + ExtraTrees + GBR) con"
    f" {len(feature_cols)} features..."
)
gkf = GroupKFold(n_splits=5)

oof_lgb = np.zeros(len(train_df))
oof_et = np.zeros(len(train_df))
oof_gbr = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_et = np.zeros(len(test_df))
test_gbr = np.zeros(len(test_df))

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
  X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
  X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

  # Modelo 1: LightGBM (L1 / MAE)
  m_lgb = lgb.LGBMRegressor(
      objective="regression_l1",
      metric="mae",
      n_estimators=1500,
      learning_rate=0.02,
      num_leaves=23,
      min_child_samples=12,
      colsample_bytree=0.75,
      subsample=0.8,
      random_state=42 + fold,
      n_jobs=-1,
      verbose=-1,
  )
  m_lgb.fit(
      X_tr,
      y_tr,
      eval_X=X_va,
      eval_y=y_va,
      callbacks=[lgb.early_stopping(60, verbose=False)],
  )
  oof_lgb[val_idx] = m_lgb.predict(X_va)
  test_lgb += m_lgb.predict(X_test) / gkf.n_splits

  # Modelo 2: ExtraTreesRegressor (MAE)
  m_et = ExtraTreesRegressor(
      n_estimators=400,
      criterion="absolute_error",
      max_depth=12,
      min_samples_leaf=3,
      random_state=42 + fold,
      n_jobs=-1,
  )
  m_et.fit(X_tr, y_tr)
  oof_et[val_idx] = m_et.predict(X_va)
  test_et += m_et.predict(X_test) / gkf.n_splits

  # Modelo 3: GradientBoostingRegressor (MAE)
  m_gbr = GradientBoostingRegressor(
      loss="absolute_error",
      n_estimators=300,
      learning_rate=0.03,
      max_depth=4,
      subsample=0.8,
      random_state=42 + fold,
  )
  m_gbr.fit(X_tr, y_tr)
  oof_gbr[val_idx] = m_gbr.predict(X_va)
  test_gbr += m_gbr.predict(X_test) / gkf.n_splits

mae_lgb = mean_absolute_error(y, oof_lgb)
mae_et = mean_absolute_error(y, oof_et)
mae_gbr = mean_absolute_error(y, oof_gbr)

oof_blend = 0.45 * oof_lgb + 0.35 * oof_et + 0.20 * oof_gbr
test_blend = 0.45 * test_lgb + 0.35 * test_et + 0.20 * test_gbr
mae_blend = mean_absolute_error(y, oof_blend)

print("\n==========================================")
print(f"MAE LightGBM:      {mae_lgb:.5f}")
print(f"MAE ExtraTrees:    {mae_et:.5f}")
print(f"MAE GradBoosting:  {mae_gbr:.5f}")
print(f"MAE ENSAMBLE OOF:  {mae_blend:.5f}")
print("==========================================")

importances = pd.Series(
    m_lgb.feature_importances_, index=feature_cols
).sort_values(ascending=False)
print("\nTop 10 features más influyentes:")
print(importances.head(10))

# ==========================================
# 6. GUARDAR SUBMISSION MEJORADO
# ==========================================
submission = pd.DataFrame({
    "hex_id": test_df["hex_id"],
    "cost_of_living": np.clip(test_blend, 0.0, 1.0),
})
submission.to_csv("submission.csv", index=False)
print("\n¡Nuevo 'submission.csv' guardado y listo para Kaggle!")