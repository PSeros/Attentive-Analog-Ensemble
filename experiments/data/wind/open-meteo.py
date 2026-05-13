import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import re
import numpy as np  # Added for trigonometric functions

# Read coordinates from CSV file
coordinates = pd.read_csv('Data/Wind/coordinates.csv')

# Extract names, latitudes and longitudes
names = coordinates['Name'].tolist()
latitudes = coordinates['Latitude'].tolist()
longitudes = coordinates['Longitude'].tolist()

# Setup Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = {
    "observations": "https://archive-api.open-meteo.com/v1/archive",
    "forecasts": "https://historical-forecast-api.open-meteo.com/v1/forecast"
}
params = {
    "latitude": latitudes,
    "longitude": longitudes,
    "start_date": "2022-01-01",
    "end_date": "2025-05-31",
    "hourly": ["wind_speed_10m", "wind_direction_10m"],
    "wind_speed_unit": "ms"
}

def sort_locations(df):
    # Extract the numeric part from the location names and sort
    locations = df.columns.tolist()
    # Extract the numeric part from the location names
    # Updated regex to match the new naming format ("Location_0" instead of "Location 0")
    location_numbers = [int(re.search(r'_\d+', loc).group()[1:]) for loc in locations]
    # Sort locations based on the numeric part
    sorted_indices = sorted(range(len(location_numbers)), key=lambda k: location_numbers[k])
    sorted_locations = [locations[i] for i in sorted_indices]
    return df[sorted_locations]

for type, url in url.items():
    request = {
        "url": url,
        "params": params,
        "output_prefix": type
    }
    url = request["url"]
    params = request["params"]
    output_prefix = request["output_prefix"]

    # Request data
    responses = openmeteo.weather_api(url, params=params)

    # List to store hourly data DataFrames
    hourly_dataframes = []

    # Process each location's response
    for i, response in enumerate(responses):
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")

        # Process hourly data
        hourly = response.Hourly()
        hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}

        # Use the name from the CSV file instead of generating a new one
        hourly_data["location"] = names[i]
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

        # Create DataFrame and add to list
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        hourly_dataframes.append(hourly_dataframe)

    # Combine all DataFrames into one
    combined_hourly_dataframe = pd.concat(hourly_dataframes, ignore_index=True)

    # Calculate u and v components
    combined_hourly_dataframe['wind_direction_rad'] = combined_hourly_dataframe['wind_direction_10m'] * (np.pi / 180)
    combined_hourly_dataframe['u_component'] = -combined_hourly_dataframe['wind_speed_10m'] * np.sin(combined_hourly_dataframe['wind_direction_rad'])
    combined_hourly_dataframe['v_component'] = -combined_hourly_dataframe['wind_speed_10m'] * np.cos(combined_hourly_dataframe['wind_direction_rad'])

    # Pivot the DataFrame to have time in rows and locations in columns for wind speed
    wind_speed_pivot = combined_hourly_dataframe.pivot(index='date', columns='location', values='wind_speed_10m')
    # Pivot the DataFrame to have time in rows and locations in columns for wind direction
    wind_direction_pivot = combined_hourly_dataframe.pivot(index='date', columns='location', values='wind_direction_10m')
    # Pivot for u and v components
    u_component_pivot = combined_hourly_dataframe.pivot(index='date', columns='location', values='u_component')
    v_component_pivot = combined_hourly_dataframe.pivot(index='date', columns='location', values='v_component')

    # Sort columns by location number
    wind_speed_pivot = sort_locations(wind_speed_pivot)
    wind_direction_pivot = sort_locations(wind_direction_pivot)
    u_component_pivot = sort_locations(u_component_pivot)
    v_component_pivot = sort_locations(v_component_pivot)

    # Save the pivoted and sorted DataFrames to CSV files
    wind_speed_pivot.to_csv(f'Data/Wind/{output_prefix}_wind_speed.csv')
    wind_direction_pivot.to_csv(f'Data/Wind/{output_prefix}_wind_direction.csv')
    u_component_pivot.to_csv(f'Data/Wind/{output_prefix}_u_component.csv')
    v_component_pivot.to_csv(f'Data/Wind/{output_prefix}_v_component.csv')

    print(f"\n{output_prefix.capitalize()} wind speed, direction, u component, and v component data have been saved to separate CSV files with time in rows and locations in columns, sorted by location.\n")
