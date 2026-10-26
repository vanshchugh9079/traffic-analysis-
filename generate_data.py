"""
Generate realistic synthetic sample accident dataset for Traffic Accident Analyzer.
All data is explicitly synthetic for academic benchmarking and demonstration purposes.
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_dataset(num_records=450, output_path="TrafficAccidentAnalyzer/data/sample_accidents.csv"):
    # Deterministic seed for reproducible testing and demos
    np.random.seed(42)
    random.seed(42)

    cities = [
        {"city": "New York", "lat": 40.7128, "lon": -74.0060},
        {"city": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"city": "Chicago", "lat": 41.8781, "lon": -87.6298},
        {"city": "Houston", "lat": 29.7604, "lon": -95.3698},
        {"city": "Phoenix", "lat": 33.4484, "lon": -112.0740},
        {"city": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
        {"city": "Dallas", "lat": 32.7767, "lon": -96.7970},
        {"city": "Austin", "lat": 30.2672, "lon": -97.7431},
        {"city": "Seattle", "lat": 47.6062, "lon": -122.3321},
        {"city": "Denver", "lat": 39.7392, "lon": -104.9903},
    ]

    city_weights = [0.18, 0.16, 0.14, 0.11, 0.08, 0.09, 0.08, 0.06, 0.05, 0.05]

    severities = ["Minor", "Serious", "Fatal"]
    severity_weights = [0.66, 0.24, 0.10]

    weather_conditions = ["Clear", "Rainy", "Overcast", "Foggy", "Snow"]
    weather_weights = [0.55, 0.25, 0.10, 0.06, 0.04]

    records = []
    base_date = datetime(2023, 1, 1)

    for i in range(1, num_records + 1):
        accident_id = f"ACC-{2023000 + i}"
        
        # Select city with slight random offset in coordinates (approx 5-15 km radius)
        city_info = random.choices(cities, weights=city_weights)[0]
        city_name = city_info["city"]
        lat = round(city_info["lat"] + np.random.normal(0, 0.04), 5)
        lon = round(city_info["lon"] + np.random.normal(0, 0.04), 5)

        # Date over 18 months
        day_offset = random.randint(0, 500)
        rec_date = base_date + timedelta(days=day_offset)
        date_str = rec_date.strftime("%Y-%m-%d")

        # Realistic rush hour time distribution (peaks at 8 AM and 18 PM)
        rand_val = random.random()
        if rand_val < 0.25:
            # Morning rush hour 7:00 - 9:59
            hour = random.randint(7, 9)
        elif rand_val < 0.65:
            # Evening rush hour 16:00 - 19:59
            hour = random.randint(16, 19)
        elif rand_val < 0.88:
            # Mid-day 10:00 - 15:59
            hour = random.randint(10, 15)
        else:
            # Night / early morning 20:00 - 6:59
            hour = random.choice([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])

        minute = random.randint(0, 59)
        time_str = f"{hour:02d}:{minute:02d}"

        # Weather and correlated road condition
        weather = random.choices(weather_conditions, weights=weather_weights)[0]
        if weather == "Rainy":
            road_surface = random.choices(["Wet", "Flooded", "Dry"], weights=[0.85, 0.10, 0.05])[0]
        elif weather == "Snow":
            road_surface = random.choices(["Icy", "Snow", "Wet"], weights=[0.60, 0.30, 0.10])[0]
        elif weather == "Foggy":
            road_surface = random.choices(["Damp", "Wet", "Dry"], weights=[0.50, 0.30, 0.20])[0]
        else:
            road_surface = random.choices(["Dry", "Wet"], weights=[0.92, 0.08])[0]

        # Light condition based on hour
        if 6 <= hour <= 18:
            light = "Daylight"
        elif 19 <= hour <= 21 or hour == 5:
            light = random.choices(["Dusk/Dawn", "Darkness - Street Lights Lit"], weights=[0.4, 0.6])[0]
        else:
            light = random.choices(["Darkness - Street Lights Lit", "Darkness - No Lights"], weights=[0.75, 0.25])[0]

        # Severity
        # Adverse weather or darkness slightly increases severity
        if weather in ["Rainy", "Snow", "Foggy"] or "No Lights" in light:
            sev_weights = [0.55, 0.30, 0.15]
        else:
            sev_weights = severity_weights
        severity = random.choices(severities, weights=sev_weights)[0]

        # Vehicles involved (1 to 5)
        if severity == "Fatal":
            vehicles = random.choices([1, 2, 3, 4], weights=[0.20, 0.50, 0.20, 0.10])[0]
            casualties = random.choices([1, 2, 3, 4], weights=[0.50, 0.30, 0.15, 0.05])[0]
        elif severity == "Serious":
            vehicles = random.choices([1, 2, 3], weights=[0.30, 0.55, 0.15])[0]
            casualties = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10])[0]
        else:
            vehicles = random.choices([1, 2, 3], weights=[0.45, 0.50, 0.05])[0]
            casualties = random.choices([0, 1, 2], weights=[0.60, 0.35, 0.05])[0]

        records.append({
            "Accident_ID": accident_id,
            "Date": date_str,
            "Time": time_str,
            "City": city_name,
            "Severity": severity,
            "Weather": weather,
            "Road_Surface": road_surface,
            "Light_Condition": light,
            "Vehicles_Involved": vehicles,
            "Casualties": casualties,
            "Latitude": lat,
            "Longitude": lon
        })

    # Add a few subtle missing values or duplicate row to test cleaning robustness
    df = pd.DataFrame(records)
    
    # Introduce 3 duplicates intentionally for cleaning verification
    duplicates = df.iloc[[12, 45, 88]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Introduce a couple of missing fields in weather/road_surface
    df.loc[25, "Weather"] = None
    df.loc[60, "Road_Surface"] = np.nan
    df.loc[110, "Light_Condition"] = None

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sample records successfully at {output_path}")

if __name__ == "__main__":
    generate_sample_dataset()
