from A2E.pipeline.base import *

class A2EPipeline(BaseTrainingPipeline):
    def __init__(self, *args, **kwargs):
        """
        Specialised TrainingPipeline used to supply a dataset for training A2E.

        Args:
            forecasts: Historical forecast data
            observations: Historical observation data
            config: model configuration
            test_size: Fraction of data to use for testing
        """
        super().__init__(*args, **kwargs)
        self.num_samples = int(self.num_timesteps - self.config.lookback - self.config.time_to_target)
        self.test_size = int(tf.math.ceil(self.num_samples * self.test_size_fraction))
        self.train_size = int(self.num_samples - self.test_size)

        # Run pipeline setup
        self._validate_inputs()
        self._prepare_dataset()

    def _validate_inputs(self):
        """A2E specific assertions"""
        super()._validate_inputs()
        tf.debugging.assert_equal(
            tf.rank(self.forecasts),
            2,
            message="Forecasts must be 2D for A2E models"
        )
        tf.debugging.assert_equal(
            tf.rank(self.observations),
            2,
            message="Observations must be 2D for A2E models"
        )

    def _create_dataset(self, f_data, a_data) -> tf.data.Dataset:
        """Creates an index-based sliding window dataset for training A2E."""
        ds = tf.data.Dataset.range(tf.cast(self.num_samples, tf.int64)) # has to be tf.in64

        def extract_sample(i):
            i = tf.cast(i, tf.int32)
            x_h = f_data[i: i + self.config.lookback]
            y_h = a_data[i: i + self.config.lookback]

            start = i + self.config.lookback + self.config.time_to_target - self.config.seq_len
            end = i + self.config.lookback + self.config.time_to_target
            x_t = f_data[start + 1: end + 1]

            y_t = a_data[end - self.config.foresight]
            return (x_t, x_h, y_h), y_t
        return ds.map(extract_sample, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
