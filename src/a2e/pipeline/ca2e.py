from a2e.pipeline.base import *

class CA2EPipeline(BaseTrainingPipeline):
    def __init__(self, *args, **kwargs):
        """
        Specialised TrainingPipeline used to supply a dataset for training CA2E.

        Args:
            forecasts: Historical forecast data
            observations: Historical observation data
            config: model configuration
            test_size: Fraction of data to use for testing
        """
        super().__init__(*args, **kwargs)
        self.num_samples_per_location = int(self.num_timesteps - (self.config.lookback + self.config.time_to_target))
        self.num_samples = self.num_samples_per_location * self.config.max_locations
        self.test_size = int(tf.math.ceil(self.num_samples * self.test_size_fraction))
        self.train_size = int(self.num_samples - self.test_size)

        # Run pipeline setup
        self._validate_inputs()
        self._prepare_dataset()

    def _validate_inputs(self):
        """CA2E specific assertions"""
        super()._validate_inputs()
        tf.debugging.assert_equal(
            tf.rank(self.forecasts),
            3,
            message="Forecasts must be 3D for CA2E models"
        )
        tf.debugging.assert_equal(
            tf.rank(self.observations),
            3,
            message="Observations must be 3D for CA2E models"
        )
        tf.debugging.assert_equal(
            tf.shape(self.forecasts)[1],
            self.config.max_locations,
            message="Second dimension of forecasts must match number of max_locations"
        )
        tf.debugging.assert_equal(
            tf.shape(self.observations)[1],
            self.config.max_locations,
            message="Second dimension of observations must match number of max_locations"
        )

    def _extract_sample(self, f_data, a_data, sample):
        """
        CA2E uses the timesteps of multiple locations subsequently.
        Sampling logic goes first over all locations before shifting to the next time step:
        sample = i * max_locations + loc, i.e.
        loc=0, i=0; loc=1, i=0 ... loc=total_loc, i=0; loc=0, i=1; loc=1, i=1 ... loc=total_loc, i=sample_per_loc
        """
        sample = tf.cast(sample, tf.int32)
        i = sample // self.config.max_locations
        loc = sample % self.config.max_locations

        # Extract historical data: the forecasts are shifted by `foresight` against the values, so that the
        # window ending in row r carries a_data[r - foresight], the value at its target time.
        fs = self.config.foresight
        X_h = f_data[i + fs: i + self.config.lookback + fs, loc]
        Y_h = a_data[i: i + self.config.lookback, loc]

        # Extract current sequence
        start = i + self.config.lookback + self.config.time_to_target - self.config.seq_len
        end = i + self.config.lookback + self.config.time_to_target
        X_t = f_data[start + 1: end + 1, loc]

        # Target
        y_t = a_data[end - self.config.foresight, loc]
        return (X_t, X_h, Y_h, loc), y_t
