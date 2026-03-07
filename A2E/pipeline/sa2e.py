from A2E.pipeline.base import *

class SA2EPipeline(BaseTrainingPipeline):
    def __init__(self, *args, **kwargs):
        """
        Specialised TrainingPipeline used to supply a dataset for training SA2E.

        Args:
            forecasts (np.ndarray): Historical forecast data.
            observations (np.ndarray): Historical observation data.
            config (ModelConfig): model configuration.
            test_size (int): Fraction of data to use for testing
        """
        super().__init__(*args, **kwargs)
        self.num_samples = int(self.num_timesteps - self.config.lookback - self.config.time_to_target)
        self.test_size = int(tf.math.ceil(self.num_samples * self.test_size_fraction))
        self.train_size = int(self.num_samples - self.test_size)

        self._validate_inputs()
        self._prepare_dataset()

    def _validate_inputs(self):
        """SA2E specific assertions"""
        super()._validate_inputs()
        tf.debugging.assert_equal(
            tf.rank(self.forecasts),
            3,
            message="Forecasts must be 3D for SA2E models"
        )
        tf.debugging.assert_equal(
            tf.rank(self.observations),
            2,
            message="Observations must be 2D for SA2E models"
        )
        tf.debugging.assert_equal(
            tf.shape(self.forecasts)[1],
            self.config.d_loc,
            message="Second dimension of forecasts must match number of d_loc"
        )

    def _create_dataset(self, f_data, a_data):
        """Creates an index-based sliding window dataset for training SA2E."""
        ds = tf.data.Dataset.range(tf.cast(self.num_samples, tf.int64))

        def extract_sample(i):
            i = tf.cast(i, tf.int32)
            X_h = f_data[i: i + self.config.lookback]
            Y_h = a_data[i: i + self.config.lookback]

            start = i + self.config.lookback + self.config.time_to_target - self.config.seq_len
            end = i + self.config.lookback + self.config.time_to_target
            X_t = f_data[start + 1: end + 1]

            y_t = a_data[end - self.config.foresight]
            return (X_t, X_h, Y_h), y_t
        return ds.map(extract_sample, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
