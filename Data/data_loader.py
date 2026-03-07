import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import re
import os

class WindDataLoader:
    def __init__(self, data_dir='Data/Wind/'):
        """
        Initialize the WindDataLoader with the directory containing the data files.

        Parameters:
        -----------
        data_dir : str, optional
            The directory where the data files are stored (default is 'Data/Wind/').
        """
        self.data_dir = data_dir
        self.observation_data = None
        self.forecast_data = None

    def sort_locations(self, df):
        """
        Sort the DataFrame columns based on the numeric part of the location names.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame whose columns need to be sorted.

        Returns:
        --------
        pandas.DataFrame
            The DataFrame with sorted columns.
        """
        locations = df.columns.tolist()
        location_numbers = [int(re.search(r'_\d+', loc).group()[1:]) for loc in locations]
        sorted_indices = sorted(range(len(location_numbers)), key=lambda k: location_numbers[k])
        sorted_locations = [locations[i] for i in sorted_indices]
        return df[sorted_locations]

    def check_alignment(self, *dfs, data_type):
        """
        Check if dates and locations are aligned across multiple DataFrames.

        Parameters:
        -----------
        *dfs : pandas.DataFrame
            The DataFrames to check for alignment.
        data_type : str
            The type of data (e.g., 'observations' or 'forecasts').

        Raises:
        -------
        ValueError
            If dates or locations are not aligned.
        """
        if len(dfs) < 2:
            return  # No alignment check needed for single DataFrame
        # Check indices (dates)
        first_index = dfs[0].index
        for df in dfs[1:]:
            if not (first_index == df.index).all():
                raise ValueError(f"Dates are not aligned across components for {data_type}")
        # Check columns (locations)
        first_columns = dfs[0].columns
        for df in dfs[1:]:
            if not (first_columns == df.columns).all():
                raise ValueError(f"Locations are not aligned across components for {data_type}")

    def check_nan_inf(self, df, component, data_type):
        """
        Check for NaN or Inf values in a DataFrame.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to check.
        component : str
            The component name (for error messages).
        data_type : str
            The type of data (e.g., 'observations' or 'forecasts').
        """
        if df.isna().any().any():
            print(f"Warning: NaN values found in {component} component for {data_type}")
        if np.isinf(df.values).any():
            print(f"Warning: Inf values found in {component} component for {data_type}")

    def load_components(self, data_type, components):
        """
        Load the specified components for a given data type.

        Parameters:
        -----------
        data_type : str
            The type of data to load ('observations' or 'forecasts').
        components : list of str
            List of components to load ('u', 'v', 'total').

        Returns:
        --------
        dict of pandas.DataFrame
            A dictionary mapping component names to their respective DataFrames.
        """
        result = {}
        if data_type == 'observations':
            prefix = 'observations'
        elif data_type == 'forecasts':
            prefix = 'forecasts'
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        for comp in components:
            if comp == 'u':
                path = os.path.join(self.data_dir, f'{prefix}_u_component.csv')
            elif comp == 'v':
                path = os.path.join(self.data_dir, f'{prefix}_v_component.csv')
            elif comp == 'total':
                path = os.path.join(self.data_dir, f'{prefix}_wind_speed.csv')
            else:
                raise ValueError(f"Unknown component: {comp}")

            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = self.sort_locations(df)
            result[comp] = df

        return result

    def load_coordinates(self):
        """
        Load coordinates from a file in the data directory.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the coordinates for each location.
        """
        coordinates_path = os.path.join(self.data_dir, 'coordinates.csv')

        # Check if the coordinates file exists
        if not os.path.exists(coordinates_path):
            raise FileNotFoundError(f"No coordinates file found at {coordinates_path}")

        # Load coordinates from file
        coordinates_df = pd.read_csv(coordinates_path, index_col=0)

        return coordinates_df

    def get_data(self, data_type, components=None):
        """
        Load and preprocess data for a given data type.

        Parameters:
        -----------
        data_type : str
            The type of data to load ('observations' or 'forecasts').
        components : list of str, optional
            List of components to load. For observations, only 'total' is valid.
            For forecasts, can include 'u', 'v', 'total'. Defaults to ['u', 'v'] for forecasts
            and ['total'] for observations.

        Returns:
        --------
        numpy.ndarray
            An array of shape (time, location, n) where n is the number of components requested.
        """
        if components is None:
            if data_type == 'observations':
                components = ['total']
            else:  # forecasts
                components = ['u', 'v']

        # Check if components are valid
        valid_components = {'u', 'v', 'total'}
        if data_type == 'observations':
            # For observations, typically all components are available, but we'll check
            pass
        for comp in components:
            if comp not in valid_components:
                raise ValueError(f"Invalid component '{comp}'. Valid components are: {valid_components}")

        # Load the required components
        component_dfs = self.load_components(data_type, components)

        # Check alignment between all loaded components
        if len(component_dfs) > 1:
            self.check_alignment(*component_dfs.values(), data_type=data_type)

        # Check for NaN/Inf in each component
        for comp, df in component_dfs.items():
            self.check_nan_inf(df, comp, data_type)

        # Prepare the list of arrays in the order specified by the user
        component_arrays = []
        for comp in components:
            component_arrays.append(component_dfs[comp].values)

        # Stack the arrays along the last dimension
        if len(component_arrays) == 1:
            # Reshape to (time, location, 1) if only one component is requested
            result_array = component_arrays[0].reshape(component_arrays[0].shape[0], component_arrays[0].shape[1], 1)
        else:
            result_array = np.stack(component_arrays, axis=-1)

        return result_array

    def get_all_data(self, obs_components=None, fcst_components=None):
        """
        Load and preprocess both observations and forecast data.

        Parameters:
        -----------
        obs_components : list of str, optional
            Components to load for observations. Defaults to ['total'].
        fcst_components : list of str, optional
            Components to load for forecasts. Defaults to ['u', 'v'].

        Returns:
        --------
        tuple of numpy.ndarray
            A tuple containing two numpy arrays: one for observation data and one for forecast data.
        """
        self.observation_data = self.get_data('observations', components=obs_components)
        self.forecast_data = self.get_data('forecasts', components=fcst_components)
        return self.observation_data, self.forecast_data

    def find_nearest_locations(self, n):
        """
        Find the n nearest locations for each location based on their coordinates.

        Parameters:
        -----------
        n : int
            The number of nearest neighbors to find for each location.

        Returns:
        --------
        numpy.ndarray
            An array of shape (locations, n+1) where for each location, the first element
            is the location itself followed by the n nearest neighbors sorted by distance.

        Raises:
        -------
        ValueError
            If n is greater than or equal to the total number of locations.
        """
        # Load coordinates
        coordinates = self.load_coordinates().values

        # Extract coordinates as a numpy array
        num_locations = len(coordinates)

        # Check if n is valid
        assert num_locations >= n, ValueError(f"n ({n}) must be less than the total number of locations ({num_locations})")

        # Calculate pairwise Euclidean distances
        distances = euclidean_distances(coordinates)

        # Get the indices that would sort each row of the distance matrix
        sorted_indices = np.argsort(distances, axis=1)

        # Take the first n+1 indices from each row
        nearest_indices = sorted_indices[:, :n+1]

        return nearest_indices

if __name__ == "__main__":
    wind_data_loader = WindDataLoader()
    n_nearest = 3  # Find the 3 nearest locations for each location
    nearest_locations = wind_data_loader.find_nearest_locations(n_nearest)
    print("Nearest locations array shape:", nearest_locations.shape)
    print(nearest_locations)


# if __name__ == "__main__":
#     data_loader = WindDataLoader()
#     # Example usage with new parameters:
#     observation_data, forecast_data = data_loader.get_all_data(
#         obs_components=['total'],  # Only speed for observations
#         fcst_components=['u', 'v', 'total']  # u, v, and wind speed for forecasts
#     )
#
#     print("Shape of observation data:", observation_data.shape)
#     print("Shape of forecast data:", forecast_data.shape)
