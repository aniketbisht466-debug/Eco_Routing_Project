import streamlit as st
import networkx as nx
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium
import os

# ReportLab Components for Corporate PDF Engine
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="AI Eco-Routing Engine", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('fuel_model.pkl')

try:
    model = load_model()
except:
    st.error("❌ 'fuel_model.pkl' nahi mili! Pehle Phase 1 run karein.")
    st.stop()

st.title("🌿 AI-Powered Eco-Routing Logistics Engine")
st.markdown("Automating corporate ESG targets by minimizing fleet carbon footprints using structural Machine Learning.")

# Sidebar Controls
st.sidebar.header("🚚 Fleet Parameters")
payload = st.sidebar.slider("Truck Payload Weight (kg)", 500, 5000, 4500, step=100) # Default setup to high load to show divergence

st.sidebar.header("🚦 Real-time Conditions")
traffic_condition = st.sidebar.slider("City Traffic Congestion Index", 1.0, 5.0, 4.2, step=0.1)
elevation_profile = st.sidebar.slider("Terrain Hill Slope (%)", -10.0, 15.0, 12.0, step=0.5)

# Graph Framework
G = nx.Graph()
locations = {
    "Warehouse": [28.6139, 77.2090],
    "Customer_A": [28.6250, 77.2200],
    "Customer_B": [28.6000, 77.2300],
    "Customer_C": [28.6350, 77.2500],
    "Customer_D": [28.5900, 77.2600]
}

for loc in locations:
    G.add_node(loc)

# Strategic routes arrangement where hills alter cost weights significantly
roads = [
    ("Warehouse", "Customer_A", {"distance": 5.2, "traffic_mult": 1.5, "elev_mult": 2.0}),
    ("Warehouse", "Customer_B", {"distance": 8.1, "traffic_mult": 0.5, "elev_mult": 0.2}),
    ("Customer_A", "Customer_C", {"distance": 3.4, "traffic_mult": 2.0, "elev_mult": 3.0}),
    ("Customer_B", "Customer_C", {"distance": 4.1, "traffic_mult": 0.4, "elev_mult": 0.1}),
    ("Customer_C", "Customer_D", {"distance": 4.5, "traffic_mult": 1.0, "elev_mult": 1.0}),
    ("Customer_A", "Customer_D", {"distance": 9.0, "traffic_mult": 0.8, "elev_mult": 0.5}),
]

for u, v, data in roads:
    # Simulating road variations dynamically based on slider values
    eff_traffic = min(5.0, max(1.0, traffic_condition * data['traffic_mult']))
    eff_elev = elevation_profile * data['elev_mult']
    
    input_data = np.array([[data['distance'], payload, eff_traffic, eff_elev]])
    predicted_fuel = float(model.predict(input_data)[0])
    
    G.add_edge(u, v, distance_cost=data['distance'], carbon_cost=predicted_fuel)

col1, col2 = st.columns(2)
with col1:
    start_point = st.selectbox("Select Start Point", list(locations.keys()), index=0)
with col2:
    end_point = st.selectbox("Select Destination", list(locations.keys()), index=3)

if start_point != end_point:
    standard_path = nx.shortest_path(G, source=start_point, target=end_point, weight='distance_cost')
    standard_dist = nx.shortest_path_length(G, source=start_point, target=end_point, weight='distance_cost')
    
    eco_path = nx.shortest_path(G, source=start_point, target=end_point, weight='carbon_cost')
    eco_fuel = nx.shortest_path_length(G, source=start_point, target=end_point, weight='carbon_cost')
    
    standard_fuel = sum(G[standard_path[i]][standard_path[i+1]]['carbon_cost'] for i in range(len(standard_path)-1))
    savings = ((standard_fuel - eco_fuel) / standard_fuel) * 100 if standard_fuel > eco_fuel else 0.0
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("🛣️ Shortest Distance", f"{standard_dist:.2f} km")
    kpi2.metric("🌱 Eco Path Fuel Expected", f"{eco_fuel:.2f} L")
    kpi3.metric("📉 Carbon Saved Score", f"{savings:.2f}%", delta=f"-{savings:.1f}% Emissions" if savings > 0 else None)

    # -----------------------------------------------------------------
    # EXPORTING MECHANISM: REPORTLAB AUTOMATED PDF BUILDER
    # -----------------------------------------------------------------
    def generate_pdf_report(filename, s_path, e_path, dist, s_fuel, e_fuel, pct_saved, p_load):
        doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40, title="Corporate ESG Compliance Log")
        styles = getSampleStyleSheet()
        story = []
        
        # Header styles
        title_p = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#1E3D2F'))
        body_p = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=10, leading=14)
        
        story.append(Paragraph("<b>GREEN LOGISTICS ENGINE: ESG AUDIT REPORT</b>", title_p))
        story.append(Spacer(1, 15))
        
        meta_text = f"<b>Operational Cargo Matrix:</b> Fleet Payload: {p_load} kg | Evaluation Routing Status: Validated<br/>" \
                    f"<b>Optimized Route Architecture:</b> {' ➔ '.join(e_path)}"
        story.append(Paragraph(meta_text, body_p))
        story.append(Spacer(1, 15))
        
        # Data Matrix
        report_data = [
            [Paragraph("<b>Performance Metric</b>", body_p), Paragraph("<b>Value Logged</b>", body_p)],
            [Paragraph("Total Physical Mileage Travelled", body_p), Paragraph(f"{dist:.2f} km", body_p)],
            [Paragraph("Standard Route Projected Consumption", body_p), Paragraph(f"{s_fuel:.2f} Liters", body_p)],
            [Paragraph("AI Optimized Route Consumption", body_p), Paragraph(f"{eco_fuel:.2f} Liters", body_p)],
            [Paragraph("Net Corporate Carbon Footprint Reduction", body_p), Paragraph(f"<b>{pct_saved:.2f}% Fuel Saved</b>", body_p)]
        ]
        
        t = Table(report_data, colWidths=[240, 240])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.HexColor('#1E3D2F')),
            ('TEXTCOLOR', (0,0), (1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        doc.build(story)

    # Trigger Button for the User
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Enterprise Actions")
    if st.sidebar.button("Generate Sustainability Audit PDF"):
        pdf_name = "Fleet_Carbon_Audit_Report.pdf"
        generate_pdf_report(pdf_name, standard_path, eco_path, standard_dist, standard_fuel, eco_fuel, savings, payload)
        
        with open(pdf_name, "rb") as f:
            st.sidebar.download_button(
                label="📥 Download Official ESG Report",
                data=f,
                file_name="Corporate_Sustainability_Audit.pdf",
                mime="application/pdf"
            )
        st.sidebar.success("✅ Audit PDF generated successfully!")

    # Live Visual Maps Rendering
    st.subheader("🗺️ Live Route Spatial Optimization Tracker")
    m = folium.Map(location=locations[start_point], zoom_start=13, tiles="OpenStreetMap")
    
    for name, coords in locations.items():
        color = "green" if name in [start_point, end_point] else "blue"
        folium.Marker(location=coords, popup=name, icon=folium.Icon(color=color, icon="info-sign")).add_to(m)

    standard_coords = [locations[node] for node in standard_path]
    eco_coords = [locations[node] for node in eco_path]

    folium.PolyLine(standard_coords, color="red", weight=4, opacity=0.6, tooltip="Standard Fast Route").add_to(m)
    folium.PolyLine(eco_coords, color="green", weight=6, opacity=0.8, tooltip="AI Eco-Friendly Route").add_to(m)

    st_folium(m, width=900, height=450)
