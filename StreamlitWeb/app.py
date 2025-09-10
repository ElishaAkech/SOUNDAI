import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the MLP class (must match the saved model's architecture)
class MLPNoisePredictor(nn.Module):
    def __init__(self, input_size):
        super(MLPNoisePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 25)  # First hidden layer
        self.fc2 = nn.Linear(25, 50)         # Second hidden layer
        self.fc3 = nn.Linear(50, 1)          # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output for regression
        return x

# Load the saved model
@st.cache_resource
def load_model():
    input_size = 7  # Number of features: motorcycles, light, medium, heavy, speed, lanes, flow_type
    model = MLPNoisePredictor(input_size)
    model.load_state_dict(torch.load('StreamlitWeb/models/new_MLP_noise_predictor_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Load data for location info (e.g., lanes, average values)
file_path = 'data/FINAL DATA.xlsx'  
df_leq = pd.read_excel(file_path, sheet_name='Noise Leq Data')
df_speed = pd.read_excel(file_path, sheet_name='SPEED')
df_pcu = pd.read_excel(file_path, sheet_name='PCU')
df_pcu_conv = pd.read_excel(file_path, sheet_name='PCU Conversion')
places = df_leq['Place'].dropna().unique()

# Dictionary for lanes (from previous code)
lanes_dict = {
    'Around Arya School': 2, 'Around Baba Dogo Rd': 2, 'Around Junction Mall': 4, 'Around Langata Hospital': 4,
    'Around MMU': 4, 'BBS Eastleigh': 4, 'Bee Centre': 2, 'Close to Uhuru Park': 2, 'Davis&Shirtliff Kangundo Rd': 2,
    'ICD Road': 4, 'Imaara Mall': 4, 'Jevanjee': 2, 'Jogoo Road': 4, 'Kangemi': 4, 'Karen C School': 2,
    'Kawangware': 2, 'KCB Utawala Eastern Bypass': 4, 'KFC Embakasi': 4, 'Kiambu Road': 2, 'Kiambu Road 2': 2,
    'Kinoo': 2, 'Langata Link Road': 4, 'Likoni Road': 4, 'Makongeni Shopping Centre Ruai': 2, 'Ngong Road': 4,
    'Northern Bypass': 2, 'Nyayo Langata': 4, 'Ola Energy Waiyaki Way': 8, 'Opp. KU Hospital': 2,
    'Quality Meat Packers': 2, 'Raila Odinga Road Next to Total': 4, 'Ruaka': 2, 'Runda': 2, 'Southern Bypass 1': 4,
    'Southern Bypass 2': 4, 'Thika Road 1': 8, 'Thika Road 2': 8, 'Thika Road (Pangani)': 8, 'Thome': 2,
    'Total Energies Outering': 8, 'Winners Chapel (Likoni Road)': 4, 'Junction Mall': 4, 'Arya (Ngara)': 2,
    'Around Baba Dogo Road': 2
}

# Hour bands
hours = ['6-7AM', '7-8AM', '8-9AM', '9-10AM', '10-11AM', '11-12PM', '12-1PM', '1-2PM', '2-3PM', '3-4PM', '4-5PM', '5-6PM']

# Fit a dummy scaler (in practice, load or fit on training data features)
# For demonstration, we assume the scaler is fitted on similar data; replace with actual scaler if saved
@st.cache_resource
def get_scaler():
    # Dummy data for scaler fitting (replace with actual training data)
    dummy_data = np.array([
        [10, 50, 20, 5, 40, 4, 1],  # Example row
        [20, 100, 40, 10, 50, 2, 2]
    ])
    scaler = StandardScaler()
    scaler.fit(dummy_data)
    return scaler

scaler = get_scaler()

# Streamlit app
st.title("Noise Level Prediction App")

# Sidebar for input
st.sidebar.header("Input Parameters")
mode = st.sidebar.radio("Mode", ("Manual Input", "Location-Based"))

if mode == "Manual Input":
    st.sidebar.subheader("Enter Values")
    motorcycles = st.sidebar.number_input("Motorcycles", min_value=0.0, value=10.0)
    light = st.sidebar.number_input("Light Vehicles", min_value=0.0, value=50.0)
    medium = st.sidebar.number_input("Medium Vehicles", min_value=0.0, value=20.0)
    heavy = st.sidebar.number_input("Heavy Vehicles", min_value=0.0, value=5.0)
    speed = st.sidebar.number_input("Speed (km/h)", min_value=0.0, value=40.0)
    lanes = st.sidebar.number_input("Lanes", min_value=1, max_value=8, value=4)
    flow_type = st.sidebar.selectbox("Flow Type", [0, 1, 2])  # 0: Congested, 1: Moderate, 2: Free-flow

    if st.sidebar.button("Predict Noise Level"):
        # Prepare input
        input_data = np.array([[motorcycles, light, medium, heavy, speed, lanes, flow_type]])
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            predicted_leq = model(input_tensor).item()
        
        st.success(f"Predicted Noise Level (Leq): {predicted_leq:.2f} dBA")

else:  # Location-Based
    st.sidebar.subheader("Select Location and Time")
    selected_place = st.sidebar.selectbox("Location", places)
    selected_hour = st.sidebar.selectbox("Time Band", hours)
    
    if st.sidebar.button("Get Estimates"):
        # Get index for selected place and hour
        place_idx = list(places).index(selected_place)
        hour_idx = hours.index(selected_hour)
        
        # Fetch historical/estimated values (from sheets)
        estimated_speed = df_speed.iloc[place_idx, hour_idx + 2] if pd.notna(df_speed.iloc[place_idx, hour_idx + 2]) else 40.0
        estimated_pcu = df_pcu.iloc[place_idx, hour_idx + 2] if pd.notna(df_pcu.iloc[place_idx, hour_idx + 2]) else 1000.0
        lanes = lanes_dict.get(selected_place, 4)
        
        # Estimate vehicle counts (using proportions as in data processing)
        place_conv = df_pcu_conv[df_pcu_conv['Place'].str.strip().str.upper() == selected_place.upper()]
        if not place_conv.empty:
            conv_row = place_conv.iloc[0, 3:].values
            vehicle_cols = df_pcu_conv.columns[3:]
            
            motorcycles = conv_row[list(vehicle_cols).index('MotorCycle')] if 'MotorCycle' in vehicle_cols else 0
            light_vehicles = sum(conv_row[list(vehicle_cols).index(col)] for col in ['Private car', 'Pickup', 'SUV'] if col in vehicle_cols)
            medium_vehicles = sum(conv_row[list(vehicle_cols).index(col)] for col in ['Buses', 'Light trucks', 'Psv'] if col in vehicle_cols)
            heavy_vehicles = sum(conv_row[list(vehicle_cols).index(col)] for col in ['Medium trucks', 'Heavy trucks'] if col in vehicle_cols)
            
            total_vehicles = motorcycles + light_vehicles + medium_vehicles + heavy_vehicles
            average_pcu_per_vehicle = estimated_pcu / total_vehicles if total_vehicles > 0 else 0
            
            motorcycles = (estimated_pcu / average_pcu_per_vehicle) * (motorcycles / total_vehicles) if total_vehicles > 0 else 0
            light = (estimated_pcu / average_pcu_per_vehicle) * (light_vehicles / total_vehicles) if total_vehicles > 0 else 0
            medium = (estimated_pcu / average_pcu_per_vehicle) * (medium_vehicles / total_vehicles) if total_vehicles > 0 else 0
            heavy = (estimated_pcu / average_pcu_per_vehicle) * (heavy_vehicles / total_vehicles) if total_vehicles > 0 else 0
            
            flow_type = 0 if estimated_speed < 20 else 1 if estimated_speed < 35 else 2
        else:
            motorcycles = light = medium = heavy = 0
            flow_type = 1  # Default
            
        # Prepare input
        input_data = np.array([[motorcycles, light, medium, heavy, estimated_speed, lanes, flow_type]])
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            predicted_leq = model(input_tensor).item()
        
        st.success(f"Predicted Noise Level (Leq): {predicted_leq:.2f} dBA")
        st.info(f"Estimated Speed: {estimated_speed:.2f} km/h")
        st.info(f"Estimated Vehicle Count: Motorcycles={motorcycles:.0f}, Light={light:.0f}, Medium={medium:.0f}, Heavy={heavy:.0f}")