from a2e.pipeline.fast_a2e import *

class FastCA2EPipeline(FastA2EPipeline):
    def __init__(self, *args, **kwargs):
        """
        Specialised TrainingPipeline used to supply a dataset for training FastCA2E.
        Every step contains the full series of `sites_per_step` sites and `batch_size` queries into them.
        Samples are defined as in CA2EPipeline, but split in time per location.

        Args:
            forecasts: Historical forecast data
            observations: Historical observation data
            config: model configuration
            test_size: Fraction of data to use for testing
        """
        super().__init__(*args, **kwargs)

    def _get_num_series(self) -> int:
        """One series per location"""
        return self.config.max_locations

    def _get_sites_per_step(self) -> int:
        """Number of site series encoded per training step"""
        return self.config.sites_per_step or 1

    def _validate_inputs(self):
        """FastCA2E specific assertions"""
        BaseTrainingPipeline._validate_inputs(self)
        tf.debugging.assert_equal(
            tf.rank(self.forecasts),
            3,
            message="Forecasts must be 3D for FastCA2E models"
        )
        tf.debugging.assert_equal(
            tf.rank(self.observations),
            3,
            message="Observations must be 3D for FastCA2E models"
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
        tf.debugging.assert_less_equal(
            self.sites_per_step,
            self.config.max_locations,
            message="sites_per_step must not exceed max_locations"
        )

    def _to_series(self, f_norm, o_norm):
        """(time, location, variable) -> (location, time, variable): one series per site"""
        return tf.transpose(f_norm, [1, 0, 2]), tf.transpose(o_norm, [1, 0, 2])

    def _inputs(self, x_s, y_s, sites, idx):
        """Model inputs of one step, including the location IDs of the sites"""
        return x_s, y_s, sites, idx
