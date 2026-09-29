import networkx as nx
import joblib
import numpy as np

# 1. Trained AI Engine ko load karein
print("🔄 Loading your AI Carbon Model...")
try:
    model = joblib.load('fuel_model.pkl')
except:
    print("❌ Error: 'fuel_model.pkl' nahi mili! Pehle Phase 1 run karein.")
    exit()

# 2. Mock City Logistics Graph Network create karein
G = nx.Graph()

# Delivery Locations (Nodes) add karein
locations = ["Warehouse", "Customer_A", "Customer_B", "Customer_C", "Customer_D"]
for loc in locations:
    G.add_node(loc)

# Roads (Edges) define karein variables ke saath: [Distance, Traffic, Elevation]
roads = [
    ("Warehouse", "Customer_A", {"distance": 5.2, "traffic": 3.5, "elevation": 40.0}),
    ("Warehouse", "Customer_B", {"distance": 8.1, "traffic": 1.2, "elevation": 5.0}),
    ("Customer_A", "Customer_C", {"distance": 3.4, "traffic": 4.8, "elevation": 110.0}),
    ("Customer_B", "Customer_C", {"distance": 6.0, "traffic": 1.5, "elevation": -10.0}),
    ("Customer_C", "Customer_D", {"distance": 4.5, "traffic": 2.0, "elevation": 15.0}),
    ("Customer_A", "Customer_D", {"distance": 9.0, "traffic": 1.1, "elevation": 0.0}),
]
G.add_edges_from(roads)

# 3. AI Powered Routing Weight Function
def calculate_eco_weights(truck_payload):
    print(f"\n🚚 Calculating eco-routes for Payload Weight: {truck_payload} kg...")
    
    for u, v, data in G.edges(data=True):
        input_features = np.array([[
            data['distance'], 
            truck_payload, 
            data['traffic'], 
            data['elevation']
        ]])
        
        # AI predict karega fuel cost
        predicted_fuel = model.predict(input_features)[0]
        G[u][v]['carbon_cost'] = predicted_fuel
        G[u][v]['distance_cost'] = data['distance']

# 4. Shortest vs Eco-Friendly Path Comparison Solver
def find_best_route(start, end, payload):
    calculate_eco_weights(payload)
    
    standard_path = nx.shortest_path(G, source=start, target=end, weight='distance_cost')
    standard_distance = nx.shortest_path_length(G, source=start, target=end, weight='distance_cost')
    
    eco_path = nx.shortest_path(G, source=start, target=end, weight='carbon_cost')
    eco_fuel = nx.shortest_path_length(G, source=start, target=end, weight='carbon_cost')
    
    standard_fuel = sum(G[standard_path[i]][standard_path[i+1]]['carbon_cost'] for i in range(len(standard_path)-1))
    
    print("\n📊 --- ROUTE OPTIMIZATION RESULT ---")
    print(f"🛣️ Standard Path (Fastest): {' ➔ '.join(standard_path)} | Distance: {standard_distance:.2f} km")
    print(f"🌱 AI Eco-Path (Greenest): {' ➔ '.join(eco_path)} | Predicted Fuel: {eco_fuel:.2f} Liters")
    
    savings = ((standard_fuel - eco_fuel) / standard_fuel) * 100 if standard_fuel > eco_fuel else 0
    print(f"📉 AI Carbon Reduction: {savings:.2f}% Fuel Saved!")

# Test run with heavy payload
find_best_route("Warehouse", "Customer_C", payload=4500)
