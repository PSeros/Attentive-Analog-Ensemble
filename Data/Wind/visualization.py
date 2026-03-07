import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from datetime import datetime, timedelta
import pytz
from scipy.spatial import distance

def find_nearest_locations(reference_index, latitudes, longitudes, n=10):
    """Find the nearest 'n' locations to the reference location."""
    ref_lat = latitudes[reference_index]
    ref_lon = longitudes[reference_index]
    distances = []
    for idx, (lat, lon) in enumerate(zip(latitudes, longitudes)):
        if idx == reference_index:
            continue  # Skip the reference location itself
        dist = distance.euclidean((ref_lat, ref_lon), (lat, lon))
        distances.append((dist, idx))
    # Sort by distance and pick the first 'n'
    distances.sort()
    nearest_indices = [idx for (dist, idx) in distances[:n]]
    return nearest_indices

def plot_wind_data_interactive(start_date, end_date, red_indices=None, blue_count=0, max_wind_speed=None):
    """
    Plot wind speed and direction over Germany for a specified date span,
    updating the same plot for each timestamp.
    Parameters:
    - start_date: str or datetime, start date of the span (inclusive)
    - end_date: str or datetime, end date of the span (inclusive)
    - red_indices: list of indices for locations to be marked in red (optional)
    - blue_count: number of nearest locations to mark in blue per red point (optional)
    - max_wind_speed: manually set max windspeed for scaling (optional)
    """
    if red_indices is None:
        red_indices = []
    try:
        # Load Germany's shape
        germany = gpd.read_file('Data/Wind/germany_shape/de.shp')
        print("Germany shapefile loaded successfully")
    except Exception as e:
        print(f"Error loading Germany shapefile: {e}")
        return

    try:
        # Load coordinates and names
        coordinates = pd.read_csv('Data/Wind/coordinates.csv')
        names = coordinates['Name'].tolist()
        latitudes = coordinates['Latitude'].tolist()
        longitudes = coordinates['Longitude'].tolist()
        print(f"Loaded {len(coordinates)} coordinate points")
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        return

    # Highlighting specific locations (red points)
    if red_indices:
        highlight_lats = [latitudes[i] for i in red_indices]
        highlight_lons = [longitudes[i] for i in red_indices]
    else:
        highlight_lats, highlight_lons = [], []

    # Find nearest locations for each red point and mark them blue
    nearest_indices = []
    if red_indices and blue_count > 0:
        for idx in red_indices:
            nearest = find_nearest_locations(idx, latitudes, longitudes, n=blue_count)
            nearest_indices.extend(nearest)
        nearest_indices = list(set(nearest_indices))  # Remove duplicates if any
        nearest_lats = [latitudes[i] for i in nearest_indices]
        nearest_lons = [longitudes[i] for i in nearest_indices]
    else:
        nearest_lats, nearest_lons = [], []

    try:
        # Load wind data (using historical data)
        wind_speed_df = pd.read_csv('Data/Wind/observations_wind_speed.csv', index_col=0, parse_dates=True)
        wind_direction_df = pd.read_csv('Data/Wind/observations_wind_direction.csv', index_col=0, parse_dates=True)
        print(f"Wind data loaded: {len(wind_speed_df)} speed records, {len(wind_direction_df)} direction records")
    except Exception as e:
        print(f"Error loading wind data: {e}")
        return

    # Make sure the index is timezone-aware (convert to UTC if needed)
    if wind_speed_df.index.tz is None:
        wind_speed_df.index = wind_speed_df.index.tz_localize('UTC')
        wind_direction_df.index = wind_direction_df.index.tz_localize('UTC')
    else:
        wind_speed_df.index = wind_speed_df.index.tz_convert('UTC')
        wind_direction_df.index = wind_direction_df.index.tz_convert('UTC')

    # Convert input dates to datetime objects with UTC timezone
    try:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        if isinstance(end_date, str):
            # Add 23:59:59 to end_date to include the entire day
            end_date = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC) + timedelta(days=1, seconds=-1)
        print(f"Date range: {start_date} to {end_date}")
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return

    # Filter the data for the date span
    filtered_dates = wind_speed_df.index[(wind_speed_df.index >= start_date) & (wind_speed_df.index <= end_date)]
    filtered_wind_speed_df = wind_speed_df.loc[filtered_dates]
    filtered_wind_direction_df = wind_direction_df.loc[filtered_dates]

    if len(filtered_dates) == 0:
        print("No data available for the selected date span.")
        print(f"Available date range: {wind_speed_df.index.min()} to {wind_speed_df.index.max()}")
        return

    print(f"Found {len(filtered_dates)} timestamps in the selected range")

    # Use manually set max wind speed if provided, otherwise calculate it
    if max_wind_speed is not None:
        global_max_speed = max_wind_speed
    else:
        global_max_speed = filtered_wind_speed_df.max().max()
        if global_max_speed == 0 or np.isnan(global_max_speed):
            global_max_speed = 10  # fallback value
    print(f"Global max wind speed: {global_max_speed:.2f} m/s")

    # Create the figure and axis BEFORE turning on interactive mode
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size as needed

    # Plot Germany's outline once
    germany.boundary.plot(ax=ax, color='black', linewidth=0.5)
    germany.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.3)

    # Set axis limits based on data bounds
    bounds = germany.total_bounds  # [minx, miny, maxx, maxy]
    ax.set_xlim(bounds[0] - 0.5, bounds[2] + 0.5)
    ax.set_ylim(bounds[1] - 0.5, bounds[3] + 0.5)

    # Plot red points if indices are provided
    if highlight_lats and highlight_lons:
        red_scatter = ax.scatter(highlight_lons, highlight_lats, color='red', s=100, label='Highlighted Locations', zorder=5)

    # Plot blue points if indices and count are provided
    if nearest_lats and nearest_lons:
        blue_scatter = ax.scatter(nearest_lons, nearest_lats, color='blue', s=100, label='Nearest Locations', zorder=5)

    # Annotate red points if indices are provided
    if highlight_lats and highlight_lons:
        for idx, (lat, lon) in enumerate(zip(highlight_lats, highlight_lons)):
            ax.text(lon, lat, str(red_indices[idx]), color='black', fontsize=8, ha='center', va='center', zorder=6)

    # Annotate blue points if indices and count are provided
    if nearest_lats and nearest_lons:
        for idx, (lat, lon) in enumerate(zip(nearest_lats, nearest_lons)):
            ax.text(lon, lat, str(nearest_indices[idx]), color='black', fontsize=8, ha='center', va='center', zorder=6)

    # Add labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Initialize with first timestamp to set up the plot properly
    first_time = filtered_dates[0]
    try:
        # Get wind data for first timestamp
        wind_speeds = filtered_wind_speed_df.loc[first_time]
        wind_directions = filtered_wind_direction_df.loc[first_time]
        # Handle missing data
        valid_mask = ~(pd.isna(wind_speeds) | pd.isna(wind_directions))
        if valid_mask.any():
            # Filter out invalid data
            valid_speeds = wind_speeds[valid_mask].values
            valid_directions = wind_directions[valid_mask].values
            valid_lats = np.array(latitudes)[valid_mask]
            valid_lons = np.array(longitudes)[valid_mask]
            # Convert wind direction to arrow direction
            arrow_directions_rad = np.deg2rad(valid_directions)
            # Calculate U and V components (standard meteorological convention)
            u_components = -valid_speeds * np.sin(arrow_directions_rad)
            v_components = -valid_speeds * np.cos(arrow_directions_rad)
            # Calculate scale factor for quiver plot
            if global_max_speed > 0:
                scale_factor = global_max_speed / 0.15
            else:
                scale_factor = 1
            # Create a Normalize object for consistent color mapping
            norm = Normalize(vmin=0, vmax=global_max_speed)
            # Create initial quiver plot
            q = ax.quiver(
                valid_lons, valid_lats, u_components, v_components,
                valid_speeds,  # color by wind speed
                cmap='viridis',
                norm=norm,
                scale=scale_factor,
                scale_units='width',
                width=0.0065,
                headwidth=3,
                headlength=5,
                pivot='middle',
                angles='xy',
            )
            # Add colorbar
            cbar = fig.colorbar(q, ax=ax, label='Wind Speed (m/s)', shrink=0.8)
            # Set initial title
            ax.set_title(f'Wind Speed and Direction in Germany\n{first_time.strftime("%Y-%m-%d %H:%M UTC")} (Frame 1/{len(filtered_dates)})')
        else:
            print(f"No valid data for first timestamp {first_time}")
            q = None
            cbar = None
    except Exception as e:
        print(f"Error initializing plot with first timestamp: {e}")
        q = None
        cbar = None

    # Turn on interactive mode AFTER initial setup
    plt.ion()

    # Loop through each timestamp in the filtered date span (starting from second if first was successful)
    start_idx = 1 if q is not None else 0
    for i, time in enumerate(filtered_dates[start_idx:], start_idx):
        try:
            # Get wind data for this timestamp
            wind_speeds = filtered_wind_speed_df.loc[time]
            wind_directions = filtered_wind_direction_df.loc[time]
            # Handle missing data
            valid_mask = ~(pd.isna(wind_speeds) | pd.isna(wind_directions))
            if not valid_mask.any():
                print(f"No valid data for timestamp {time}")
                continue
            # Filter out invalid data
            valid_speeds = wind_speeds[valid_mask].values
            valid_directions = wind_directions[valid_mask].values
            valid_lats = np.array(latitudes)[valid_mask]
            valid_lons = np.array(longitudes)[valid_mask]
            # Convert wind direction to arrow direction
            arrow_directions_rad = np.deg2rad(valid_directions)
            # Calculate U and V components (standard meteorological convention)
            u_components = -valid_speeds * np.sin(arrow_directions_rad)
            v_components = -valid_speeds * np.cos(arrow_directions_rad)
            # Calculate scale factor for quiver plot
            if global_max_speed > 0:
                scale_factor = global_max_speed / 0.15
            else:
                scale_factor = 1
            # Create a Normalize object for consistent color mapping
            norm = Normalize(vmin=0, vmax=global_max_speed)
            # Remove previous quiver plot if it exists
            if q is not None:
                q.remove()
            # Create new quiver plot with updated data
            q = ax.quiver(
                valid_lons, valid_lats, u_components, v_components,
                valid_speeds,  # color by wind speed
                cmap='viridis',
                norm=norm,
                scale=scale_factor,
                scale_units='width',
                width=0.0065,
                headwidth=3,
                headlength=5,
                pivot='middle',
                angles='xy',
            )
            # Update title with more info
            ax.set_title(f'Wind Speed and Direction in Germany\n{time.strftime("%Y-%m-%d %H:%M UTC")} (Frame {i+1}/{len(filtered_dates)})')
            # Add colorbar only if it doesn't exist
            if cbar is None:
                cbar = fig.colorbar(q, ax=ax, label='Wind Speed (m/s)', shrink=0.8)
            # Redraw the figure
            fig.canvas.draw()
            fig.canvas.flush_events()
            # Pause to allow viewing
            plt.pause(0.2)
        except Exception as e:
            print(f"Error processing timestamp {time}: {e}")
            continue

    print("Animation complete.")
    plt.show()

# Example usage with error handling
if __name__ == "__main__":
    try:
        start_date_str = "2024-05-01"
        end_date_str = "2024-06-01"
        red_indices = [77]
        blue_count = 0
        max_wind_speed = 8
        plot_wind_data_interactive(start_date_str, end_date_str, red_indices, blue_count, max_wind_speed)
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
