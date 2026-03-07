import tensorflow as tf

class AnEn(tf.keras.Model):
    """
    Analog Ensemble (AnEn) implementation in TensorFlow with integrated windowing.

    This model handles the full AnEn process with dissimilarity defined by Delle Monache et al. (2013).
    """

    def __init__(self,
                 n_analogs: int = 50,
                 temporal_window: int = 0,
                 learn_weights: bool = False,
                 name: str = "AnEn",
                 **kwargs):
        """
        Initialize AnEn model

        Args:
            n_analogs: Number of analogs to select (k)
            temporal_window: Half-width of temporal window (tilde_t)
            learn_weights: Whether feature weights are trainable
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        self.n_analogs = n_analogs
        self.temporal_window = temporal_window
        self.window_size = 2 * temporal_window + 1
        self.learn_weights = learn_weights

        # Initialize adaptable weights
        self.feature_weights = None
        self.feature_stds = None

    def build(self, input_shape):
        """
        Build model weights
        This will be called automatically by TensorFlow when the model is first used.

        Expected input_shape: (batch_size, time_steps, features) for both
        current_forecast and historical_forecasts
        """
        n_features = input_shape[0][-1]

        # Feature weights (W_i)
        self.feature_weights = self.add_weight(
            name="feature_weights",
            shape=(n_features,),
            initializer="ones",
            trainable=self.learn_weights
        )

        # Feature standard deviations (sigma_i) - computed from historical forecasts
        self.feature_stds = self.add_weight(
            name="feature_stds",
            shape=(n_features,),
            initializer="ones",
            trainable=False
        )

    def create_temporal_windows(self, data: tf.Tensor) -> tf.Tensor:
        """
        Create temporal windows from time series data.

        Args:
            data: (n_samples, features)

        Returns:
            windowed_data: (n_windows, window_size, features)
        """
        return tf.signal.frame(data, frame_length=self.window_size, frame_step=1, axis=0)

    def compute_feature_statistics(self, historical_forecasts: tf.Tensor) -> None:
        """
        Compute and update feature standard deviations from historical data

        Args:
            historical_forecasts: (n_historical, time_steps, features)
        """
        # Compute standard deviation across all windows and time steps within windows
        stds = tf.math.reduce_std(historical_forecasts, axis=0)

        # Avoid division by zero
        stds = tf.maximum(stds, 1e-8)
        self.feature_stds.assign(stds)

    def compute_dissimilarity_matrix(
            self,
            current_windows: tf.Tensor,
            historical_windows: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute weighted multivariate Euclidean distance matrix
        Implementation of equation: ||X_t, X_h|| = sum_i (W_i/sigma_i) * sqrt(sum_j (X_i,t+j - X_i,h+j)^2)

        Args:
            current_windows: (n_current_windows, window_size, features)
            historical_windows: (n_historical_windows, window_size, features)

        Returns:
            dissimilarity_matrix: (n_current_windows, n_historical_windows)
        """
        # Expand dimensions for broadcasting
        # current: (n_current, 1, window_size, features)
        # historical: (1, n_historical, window_size, features)
        current_expanded = tf.expand_dims(current_windows, axis=1)
        historical_expanded = tf.expand_dims(historical_windows, axis=0)

        # Compute squared differences: (n_current, n_historical, window_size, features)
        diff_squared = tf.square(current_expanded - historical_expanded)

        # Sum over temporal dimension (j in the equation): (n_current, n_historical, features)
        temporal_sum = tf.reduce_sum(diff_squared, axis=2)

        # Apply square root: (n_current, n_historical, features)
        temporal_sqrt = tf.sqrt(temporal_sum)

        # Apply feature weights and normalization (W_i/sigma_i): (n_current, n_historical, features)
        weights_normalized = self.feature_weights / self.feature_stds
        weighted_distances = weights_normalized * temporal_sqrt

        # Sum over features (i in the equation): (n_current, n_historical)
        dissimilarity = tf.reduce_sum(weighted_distances, axis=2)

        return dissimilarity

    def select_analogs(
            self,
            dissimilarity_matrix: tf.Tensor,
            historical_observations: tf.Tensor
    ) -> tf.Tensor:
        """
        Select top-k analogs based on dissimilarity scores

        Args:
            dissimilarity_matrix: (n_current_windows, n_historical_windows)
            historical_observations: (n_historical_samples, output_features)
            n_historical_windows: Number of windows per historical sample

        Returns:
            analog_ensemble: (n_current_windows, n_analogs, output_features)
        """
        # Get indices of top-k analogs (smallest dissimilarity)
        _, indices = tf.nn.top_k(-dissimilarity_matrix, k=self.n_analogs)

        # Gather corresponding observations
        analog_ensemble = tf.gather(historical_observations, indices)
        return analog_ensemble

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Forward pass of AnEn model with integrated windowing

        param inputs: Dictionary
            - [0] x_t: current forecast (batch_size, time_steps, features)
            - [1] x_h: historical forecasts (n_historical, time_steps, features)
            - [2] y_h: historical observations (n_historical, output_features)

        Returns:
            analog_ensemble: (n_current_windows, n_analogs, output_features)
        """
        x_t, x_h, y_h = inputs
        tf.debugging.assert_greater_equal(
            x_t.shape[0], self.window_size,
            message=f"current_forecast first dim has to be at least {self.window_size}"
        )

        # Update feature statistics from historical data
        self.compute_feature_statistics(x_h)

        # Create temporal windows
        x_t = self.create_temporal_windows(x_t)
        x_h = self.create_temporal_windows(x_h)

        if self.temporal_window:
            # Align historical observations
            y_h = y_h[self.temporal_window:-self.temporal_window]

        # Compute dissimilarity matrix using the AnEn equation
        dissimilarity = self.compute_dissimilarity_matrix(x_t, x_h)

        # Select analogs
        analog_ensemble = self.select_analogs(
            dissimilarity, y_h
        )

        # Squeeze last dim
        analog_ensemble = tf.squeeze(analog_ensemble, axis=-1)

        return analog_ensemble


if __name__ == "__main__":
    # Example usage
    from Data.data_loader import WindDataLoader
    from A2E.io import plotting
    import matplotlib.pyplot as plt

    loader = WindDataLoader()
    observations, forecasts = loader.get_all_data(
        obs_components=["total"],
        fcst_components=["total"]
    )

    key = "WindSpeed"
    forecast_type = "NWP"
    num_current_forecasts = int((forecasts.shape[0] - 8760 * 2) * 0.3)
    location = 0
    temporal_window = 1

    # Set up Data [time, location, variable]
    l_observations=observations[:,location,:]
    l_forecasts=forecasts[:,location,:]

    x_t = l_forecasts[:200]
    x_h = l_forecasts[num_current_forecasts:]
    y_t = l_observations[:200]
    y_h = l_observations[num_current_forecasts:]

    # Create AnEn model
    model = AnEn(
        n_analogs=100,
        temporal_window=temporal_window,
    )
    model.build((x_t.shape, x_h.shape, y_h.shape))

    # Get analog ensemble
    ensemble = model([x_t, x_h, y_h]).numpy()

    # align y_c with ensemble
    if temporal_window:
        y_t = y_t[temporal_window:-temporal_window]
        x_t = x_t[temporal_window:-temporal_window]

    # Ensure float64 due to a bug in plotting right now!
    y_t = y_t.astype("float64")
    x_t = x_t.astype("float64")
    ensemble = ensemble.astype("float64")

    plotting.plot_values(y_t, x_t, ensemble)
    plt.show()