import geopandas as gpd
import matplotlib.pyplot as plt
import random
from shapely.geometry import Point, MultiPolygon
import csv
from shapely.ops import unary_union

def generate_equally_distributed_coordinates(polygon, num_coordinates):
    area = polygon.area
    area_per_point = area / num_coordinates

    coordinates = []
    min_x, min_y, max_x, max_y = polygon.bounds
    points_added = 0
    attempts = 0
    while points_added < num_coordinates:
        latitude = random.uniform(min_y, max_y)
        longitude = random.uniform(min_x, max_x)
        point = Point(longitude, latitude)
        if polygon.contains(point):
            too_close = False
            for (lat, lon) in coordinates:
                existing_point = Point(lon, lat)
                if point.distance(existing_point) < (area_per_point**0.5):
                    too_close = True
                    break
            if not too_close:
                coordinates.append((latitude, longitude))
                points_added += 1
        attempts += 1
        if attempts > num_coordinates * 1000:  # Prevent infinite loops
            break
    return coordinates

# Load Germany's boundaries from the shapefile
germany = gpd.read_file('Data/Wind/germany_shape/de.shp')
germany_boundary = germany.geometry.union_all()
if isinstance(germany_boundary, MultiPolygon):
    germany_boundary = unary_union(germany.geometry)

# Generate 100 equally distributed coordinates within Germany's boundaries
equally_distributed_coordinates = generate_equally_distributed_coordinates(germany_boundary, 100)

# Create location names
location_names = [f"Location_{i}" for i in range(len(equally_distributed_coordinates))]

# Save the coordinates and location names to a CSV file
with open('Data/Wind/coordinates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Latitude', 'Longitude'])
    for i, coord in enumerate(equally_distributed_coordinates):
        writer.writerow([location_names[i], coord[0], coord[1]])

# Extract latitudes and longitudes (for plotting)
latitudes = [coord[0] for coord in equally_distributed_coordinates]
longitudes = [coord[1] for coord in equally_distributed_coordinates]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
germany.plot(ax=ax, color='lightblue', edgecolor='black')
ax.scatter(longitudes, latitudes, color='red', marker='o', label='Equally Distributed Points')
ax.set_title('Equally Distributed Coordinates within Germany')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True)
plt.legend()
plt.show()
