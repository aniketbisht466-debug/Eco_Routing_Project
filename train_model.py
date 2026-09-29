import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================================
# 1. GENERATE PHYSICS-BACKED VEHICLE SIMULATION DATA
# =====================================================================
print("⏳ Step 1: Generating synthetic driving/fleet data...")
np.random.seed(42)  # Ensures the same results every time you run it
n_samples = 5000

# Generating factors affecting fuel: distance, cargo weight, traffic, hills
distance_km = np.random.uniform(1.0, 15.0, n_samples)          
payload_kg = np.random.uniform(500, 5000, n_samples)          
traffic_index = np.random.uniform(1.0, 5.0, n_samples)         
elevation_change = np.random.uniform(-50.0, 150.0, n_samples)  

# Dynamic calculations for fuel burn (simulating real-world vehicle physics)
base_burn = distance_km * 0.15 
payload_penalty = (payload_kg / 1000) * 0.03 * distance_km
traffic_penalty = (traffic_index - 1) * 0.05 * distance_km
elevation_penalty = np.where(elevation_change > 0, (elevation_change / 100) * 0.08 * distance_km, 0)

# Final calculation + slight real-world random variation
fuel_consumed_liters = base_burn + payload_penalty + traffic_penalty + elevation_penalty + np.random.normal(0, 0.05, n_samples)
fuel_consumed_liters = np.clip(fuel_consumed_liters, 0.1, None)

# Structure into a clean spreadsheet/dataframe
df = pd.DataFrame({
    'distance_km': distance_km,
    'payload_kg': payload_kg,
    'traffic_index': traffic_index,
    'elevation_change': elevation_change,
    'fuel_consumed_liters': fuel_consumed_liters
})

# =====================================================================
# 2. TRAIN THE PREDICTIVE AI MODEL
# =====================================================================
print("⚙️ Step 2: Training the predictive AI engine on your CPU...")
X = df[['distance_km', 'payload_kg', 'traffic_index', 'elevation_change']]
y = df['fuel_consumed_liters']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimized to run extremely fast on your HP ProBook CPU
model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Calculate testing results
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# =====================================================================
# 3. VERIFY AND EXPORT FILE BINARIES
# =====================================================================
print("\n✅ Success Metrics for your LinkedIn write-up:")
print(f"   • Model Accuracy (R² Score): {r2 * 100:.2f}%")
print(f"   • Error Margin (MAE): {mae:.4f} Liters")

# Save outputs directly into your active folder
joblib.dump(model, 'fuel_model.pkl')
df.to_csv('simulated_fleet_data.csv', index=False)
print("\n💾 Saved files: 'fuel_model.pkl' and 'simulated_fleet_data.csv' are ready!")
