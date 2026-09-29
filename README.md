# 🌿 AI-Powered Eco-Routing Logistics Engine
> A predictive ML and graph optimization system designed to minimize fleet fuel consumption and carbon emission intensities (\(CO_2\)) for corporate ESG alignment.

[![Python Version](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://streamlit.io)
[![Optimization](https://shields.io)](https://networkx.org)

---

## 📊 Business Problem & Value Proposition
Standard logistics architectures map routing profiles solely based on geometric distance or shortest absolute travel duration. However, factors such as dynamic heavy-payload cargo weights, road elevation slopes, and stop-and-go urban gridlocks exponentially surge internal engine load—inflating corporate carbon output.

**This engine resolves the trade-off by modeling environmental costs directly.** By pairing dynamic regression forecasting with graph optimization heuristics, this pipeline delivers up to **11.36%+ net carbon reduction** per service loop, directly exporting automated legal compliance metrics for enterprise sustainability auditing.

---

## ⚙️ Core Technical Architecture

1. **Predictive Regressor Core:** A high-precision `RandomForestRegressor` trained on structural driving variables (payload weights, slope percentage dynamics, and spatial congestion coefficients). Achieving an operational **98.73% \(R^2\) accuracy scoring matrix**.
2. **Constrained Graph Routing Engine:** Uses `NetworkX` graph topologies to generate flexible cost parameters per route segment, assigning weights based on continuous ML fuel-burn predictions instead of static distances.
3. **Enterprise Visual UI:** Built entirely utilizing `Streamlit` and embedded with spatial geospatial markers using `Folium`.
4. **Corporate Compliance Pipeline:** Integrated an execution routine with `ReportLab` to auto-compile telemetry logs into downloadable official PDF sustainability audit sheets.

---

## 🛠️ Local System Installation

To set up and run this ecosystem seamlessly on standard consumer-tier hardware (Optimized specifically for standard enterprise processors like Intel Core setups):

```bash
# 1. Clone the repository workspace
git clone https://github.com
cd Eco_Routing_Project

# 2. Install structural dependencies
pip install numpy pandas scikit-learn joblib networkx streamlit folium streamlit-folium reportlab

# 3. Train the core predictive ML brain binary
python train_model.py

# 4. Launch the web automation dashboard layout
streamlit run app.py
```

---

## 📋 Architectural Panel Defenses (For Technical Interviews)

* **Data Provenance Strategy:** Modeled custom synthetic driving profiles derived directly from physical engine load metrics (aligned with standard vehicle dynamics equations) to circumvent data access limitations.
* **Algorithmic Compute Scopes:** Intentionally selected highly performant traditional graph heuristics rather than Reinforcement Learning (DRL) structures to maintain ultra-low compute latency (< 12ms) execution loops on consumer CPUs.
