import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Data.data_loader import WindDataLoader
# Load the data
loader = WindDataLoader()
observations, forecasts = loader.get_all_data(
    obs_components=["total"],
    fcst_components=["total"]
)
# Reshape observations to (time, locations)
observations_reshaped = observations[:, :, 0]  # shape (time, location)
num_time_steps, num_locations = observations_reshaped.shape
# Get nearest locations
nearest_locations = loader.find_nearest_locations(num_locations-1)  # shape (num_locations, n_nearest)
# Transpose observations to (locations, time)
observations_T = observations_reshaped.T  # shape (num_locations, num_time_steps)
# Get the time series of nearest neighbors for each location
nearest_ts = observations_T[nearest_locations, :]  # shape (num_locations, n_nearest, num_time_steps)
# Get the time series of each location itself
location_ts = observations_T[:, np.newaxis, :]  # shape (num_locations, 1, num_time_steps)
# Compute means along the time axis
X_mean = np.mean(location_ts, axis=2, keepdims=True)  # shape (num_locations, 1, 1)
Y_mean = np.mean(nearest_ts, axis=2, keepdims=True)   # shape (num_locations, n_nearest, 1)
# Center the data
X_centered = location_ts - X_mean  # shape (num_locations, 1, num_time_steps)
Y_centered = nearest_ts - Y_mean   # shape (num_locations, n_nearest, num_time_steps)
# Compute covariance (sum of products of centered values)
cov = np.sum(X_centered * Y_centered, axis=2)  # shape (num_locations, n_nearest)
# Compute the sum of squared differences for X and Y
X_ssd = np.sum((location_ts - X_mean) ** 2, axis=2)  # shape (num_locations, 1)
Y_ssd = np.sum((nearest_ts - Y_mean) ** 2, axis=2)   # shape (num_locations, n_nearest)
# Compute the standard deviations
X_std = np.sqrt(X_ssd)  # shape (num_locations, 1)
Y_std = np.sqrt(Y_ssd)  # shape (num_locations, n_nearest)
# Compute correlation
correlation = cov / (X_std * Y_std)
# Plot the correlogram
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, cmap='viridis', vmin=0, vmax=1)
plt.title('Correlogram: Each Location to Its Nearest Locations')
plt.xlabel('Nearest Location Index (0 is nearest, last is furthest)')
plt.ylabel('Location Index')
plt.show()