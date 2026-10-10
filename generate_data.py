"""
generate_data.py
────────────────
Generates a synthetic 5,000-row training dataset for the
AI-Powered Eco-Routing Logistics Engine (RandomForestRegressor).

Output: simulated_data.csv
Columns:
    Distance_km          – Road segment distance (2.0 – 30.0 km)
    Payload_Weight_kg    – Cargo weight          (1,000 – 8,000 kg)
    Traffic_Index        – Congestion index      (1.0 – 5.0)
    Hill_Slope_Percent   – Terrain gradient      (0.0 – 20.0 %)
    Fuel_Consumed_Liters – TARGET                (algorithmically derived)

Fuel formula (realistic approximation):
    base         = Distance_km * 0.08
    payload_adj  = (Payload_Weight_kg / 1000) * 0.12 * Distance_km
    traffic_adj  = (Traffic_Index ** 1.4) * Distance_km * 0.05
    slope_adj    = (Hill_Slope_Percent ** 1.6) * Distance_km * 0.004
    noise        = Gaussian(0, 0.25)
    fuel         = base + payload_adj + traffic_adj + slope_adj + noise
"""

import numpy as np
import pandas as pd

# -- Reproducibility ----------------------------------------------------------
SEED = 42
np.random.seed(SEED)
N    = 5_000

# -- Feature Sampling ---------------------------------------------------------
distance = np.random.uniform(2.0,   30.0,  N)   # km
payload  = np.random.uniform(1_000, 8_000, N)   # kg
traffic  = np.random.uniform(1.0,   5.0,   N)   # index
slope    = np.random.uniform(0.0,   20.0,  N)   # percent

# -- Fuel Formula -------------------------------------------------------------
base        = distance * 0.08
payload_adj = (payload / 1_000) * 0.12 * distance
traffic_adj = (traffic ** 1.4)  * distance * 0.05
slope_adj   = (slope   ** 1.6)  * distance * 0.004
noise       = np.random.normal(0, 0.25, N)

fuel = base + payload_adj + traffic_adj + slope_adj + noise
fuel = np.clip(fuel, 0.5, None)           # no negative fuel values

# -- Assemble DataFrame -------------------------------------------------------
df = pd.DataFrame({
    "Distance_km":          np.round(distance, 2),
    "Payload_Weight_kg":    np.round(payload,  1),
    "Traffic_Index":        np.round(traffic,  2),
    "Hill_Slope_Percent":   np.round(slope,    2),
    "Fuel_Consumed_Liters": np.round(fuel,     4),
})

# -- Save ---------------------------------------------------------------------
output_path = "simulated_data.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved -> {output_path}")
print(f"    Rows : {len(df):,}")
print(f"    Cols : {list(df.columns)}")
print("\n-- Sample (head 5) --")
print(df.head())
print("\n-- Fuel stats --")
print(df["Fuel_Consumed_Liters"].describe().round(3))
