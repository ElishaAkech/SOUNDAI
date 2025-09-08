import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, tan, pi

# Function to convert UTM to latitude/longitude (WGS84)
def utm_to_latlon(easting, northing, zone_number=37, northern=True):
    # Constants for WGS84
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis
    e_sq = f * (2 - f)  # eccentricity squared
    k0 = 0.9996  # scale factor
    lon0 = (zone_number - 30) * 6 - 3  # central meridian in degrees
    
    # Adjust easting for UTM false easting
    easting = easting - 500000
    
    # Convert to radians
    lon0 = radians(lon0)
    
    # Calculate meridional arc
    M = northing / k0
    e1 = (1 - sqrt(1 - e_sq)) / (1 + sqrt(1 - e_sq))
    mu = M / (a * (1 - e_sq / 4 - 3 * e_sq**2 / 64 - 5 * e_sq**3 / 256))
    
    # Footprint latitude
    phi1 = mu + (3 * e1 / 2 - 27 * e1**3 / 32) * sin(2 * mu) + \
           (21 * e1**2 / 16 - 55 * e1**4 / 32) * sin(4 * mu) + \
           (151 * e1**3 / 96) * sin(6 * mu)
    
    # Constants for latitude calculation
    C1 = e_sq * cos(phi1)**2
    T1 = tan(phi1)**2
    N1 = a / sqrt(1 - e_sq * sin(phi1)**2)
    R1 = a * (1 - e_sq) / (1 - e_sq * sin(phi1)**2)**1.5
    D = easting / (N1 * k0)
    
    # Latitude
    phi = phi1 - (N1 * tan(phi1) / R1) * (D**2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1**2 - 9 * e_sq) * D**4 / 24 +
                                          (61 + 90 * T1 + 298 * C1 + 45 * T1**2 - 252 * e_sq - 3 * C1**2) * D**6 / 720)
    lat = phi * 180 / pi
    
    # Longitude
    lon = lon0 + (D - (1 + 2 * T1 + C1) * D**3 / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1**2 + 8 * e_sq + 24 * T1**2) * D**5 / 120) / cos(phi1)
    lon = lon * 180 / pi
    
    # Apply offset for Nairobi (based on known points)
    lat_offset = -0.001
    lon_offset = 0.001
    return lat + lat_offset, lon + lon_offset

# Load data
coords_df = pd.read_excel("Coordinates.xlsx")
noise_df = pd.read_excel("FINAL DATA.xlsx", sheet_name="Noise Leq Data")

# Clean coordinates data
coords_df = coords_df[['Place', 'X-COORD', 'Y-COORD']].dropna()
coords_df.columns = ['place', 'easting', 'northing']

# Convert UTM to lat/lon
coords_df['lat'], coords_df['lon'] = zip(*coords_df.apply(lambda row: utm_to_latlon(row['easting'], row['northing']), axis=1))

# Clean noise data
noise_df = noise_df[['Place'] + [col for col in noise_df.columns if col.startswith('6-7AM') or col.startswith('7-8AM') or 
                                col.startswith('8-9AM') or col.startswith('9-10AM') or col.startswith('10-11AM') or 
                                col.startswith('11-12PM') or col.startswith('12-1PM') or col.startswith('1-2PM') or 
                                col.startswith('2-3PM') or col.startswith('3-4PM') or col.startswith('4-5PM') or 
                                col.startswith('5-6PM')]].dropna()
noise_df.columns = ['place', 'leq_6_7AM', 'leq_7_8AM', 'leq_8_9AM', 'leq_9_10AM', 'leq_10_11AM', 'leq_11_12PM', 
                    'leq_12_1PM', 'leq_1_2PM', 'leq_2_3PM', 'leq_3_4PM', 'leq_4_5PM', 'leq_5_6PM']

# Merge dataframes
merged_df = pd.merge(coords_df[['place', 'lat', 'lon']], noise_df, on='place', how='inner')

# Save to CSV
merged_df.to_csv("nairobi_leq.csv", index=False)
print("CSV file 'nairobi_leq.csv' created successfully.")