import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import time

# ==========================================
# Configuration
# ==========================================

START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

OUTPUT_FOLDER = "Weather Data"

Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# ==========================================
# 15 Main Sri Lankan Cities
# ==========================================

cities = {
    #"Colombo": (6.928843774169497, 79.8613713258848),
    #"Gampaha": (7.098033323681068, 79.99391662073124),
    #"Kalutara": (6.600835859544092, 79.96338442654711),
    #"Kandy": (7.300899327738637, 80.63268307932098),
    #"Matale": (7.493131840618484, 80.62785152309227),
    #"Nuwara Eliya": (6.962600267424384, 80.77008259796669),
    #"Galle": (6.0366680267252075, 80.21658087280191),
    #"Matara": (5.955935315290389, 80.5474189706954),
    #"Hambantota": (6.14909816700498, 81.12448109899657),
    #"Ratnapura": (6.718008092135873, 80.38619211127242),
    #"Kegalle": (7.264689349592688, 80.33881540025924),
    #"Kurunegala": (7.501960251091287, 80.36710648229555),
    #"Puttalam": (8.083890586213421, 79.82766912948505),
    #"Jaffna": (9.749500380499805, 80.00476309994767),
    #"Vavuniya": (8.812831916388358, 80.49419531464424),
    "Mannar": (9.053498209506367, 79.89163491675093),
    "Kilinochchi": (9.477449363484142, 80.36731150628948),
    "Mullaitivu": (9.365554630724302, 80.82271960199868),
    "Anuradhapura": (8.354737189477733, 80.3967108709439),
    "Polonnaruwa": (7.958319143387657, 80.99913762038891),
    "Trincomalee": (8.644891849636913, 81.22730482088811),
    "Batticaloa": (7.770587541145265, 81.70420510716693),
    "Ampara": (7.3334956619265235, 81.66879098825439),
    "Badulla": (7.0150545313817645, 81.0572623612415),
    "Monaragala": (6.92675080231918, 81.34658153418259),
}

# ==========================================
# Weather Variables
# ==========================================

hourly_variables = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation"
]

hourly_string = ",".join(hourly_variables)

# ==========================================
# Download Data
# ==========================================

for city, (lat, lon) in tqdm(cities.items()):

    print(f"\nDownloading {city}...")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&start_date={START_DATE}"
        f"&end_date={END_DATE}"
        f"&hourly={hourly_string}"
        "&timezone=Asia/Colombo"
    )

    max_retries = 5

    for attempt in range(max_retries):

        try:
            response = requests.get(url, timeout=120)

            if response.status_code == 429:
                print(response.text)
                continue

            response.raise_for_status()
            data = response.json()
            break

        except requests.exceptions.RequestException as e:
            print(f"❌ {city}: {e}")

            if attempt == max_retries - 1:
                print(f"Skipping {city}")
                data = None

    if data is None:
        continue

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as e:
        print(f"\n❌ {city}: Request failed")
        print(e)
        continue

    except ValueError:
        print(f"\n❌ {city}: Invalid JSON response")
        continue


    # Print any API error messages
    if "error" in data:
        print(f"\n❌ {city}: API returned an error")
        print(data)
        continue


    # Check if hourly data exists
    if "hourly" not in data:
        print(f"\n❌ {city}: No hourly data returned")
        print(data)
        continue


    # Everything is OK
    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    output_path = f"{OUTPUT_FOLDER}/{city}.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved -> {output_path}")

    # Small pause to be polite to API
    time.sleep(120)

print("\nAll downloads completed!")
