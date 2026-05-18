import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from ...project_paths import COORDINATES_CSV, COORDINATES_PNG, GERMANY_SHP

# Load Germany's shape
germany = gpd.read_file(GERMANY_SHP)

# Load coordinates and location IDs
coordinates = pd.read_csv(COORDINATES_CSV)
latitudes = coordinates['Latitude']
longitudes = coordinates['Longitude']

# Plot Germany's outline and coordinates
fig, ax = plt.subplots(figsize=(8, 8))
germany.boundary.plot(ax=ax, color='black', alpha=0.1, zorder=2)
germany.plot(ax=ax, color='lightgray', edgecolor='black', zorder=1)
ax.scatter(longitudes, latitudes, color='red', s=30, zorder=4)

# Define bbox properties
bbox_props = dict(boxstyle="round4", facecolor="white", alpha=0.9, edgecolor='none')

# Annotate each point with its location ID and a bbox
for loc_id, (lon, lat) in enumerate(zip(longitudes, latitudes)):
    ax.annotate(loc_id,
                (lon, lat),
                textcoords="offset points",
                xytext=(5,5),
                ha='center',
                bbox=bbox_props,
                zorder=3)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Coordinates with Location IDs')
plt.savefig(COORDINATES_PNG)
plt.show()
