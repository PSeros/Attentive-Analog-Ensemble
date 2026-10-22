from a2e.pipeline.base import *

class FastCA2EPipeline(BaseTrainingPipeline):
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
        self.num_samples_per_location = int(self.num_timesteps - (self.config.lookback + self.config.time_to_target))
        self.num_test_per_location = int(tf.math.ceil(self.num_samples_per_location * self.test_size_fraction))
        self.num_train_per_location = self.num_samples_per_location - self.num_test_per_location
        self.num_samples = self.num_samples_per_location * self.config.max_locations
        self.test_size = self.num_test_per_location * self.config.max_locations
        self.train_size = self.num_train_per_location * self.config.max_locations

        # Queries per step are spread evenly over the sites of a step
        self.sites_per_step = self.config.sites_per_step or 1
        self.queries_per_site = max(self.batch_size // self.sites_per_step, 1)

        # Run pipeline setup
        self._validate_inputs()
        self._prepare_dataset()

    def _validate_inputs(self):
        """FastCA2E specific assertions"""
        super()._validate_inputs()
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

    def _prepare_dataset(self) -> None:
        """Normalize data and create train and test datasets. Steps are already batched."""
        f_norm, o_norm = self._normalize()

        # (time, location, variable) -> (location, time, variable): one series per site
        f_sites = tf.transpose(f_norm, [1, 0, 2])
        o_sites = tf.transpose(o_norm, [1, 0, 2])

        self.train_dataset = self._create_dataset(f_sites, o_sites)
        self.test_dataset = self._create_test_dataset(f_sites, o_sites)

    def _create_step(self, f_sites, o_sites, sites, i):
        """
        Builds one step from the series of the given sites and sample indices i (sites, queries_per_site).
        Sample i is the same as in CA2EPipeline: lookback f[i: i+lookback], query window ending at
        i+lookback+time_to_target. Indices refer to encoded steps: z[j] <-> window ending at j+seq_len-1.
        """
        x_s = tf.gather(f_sites, sites)                                  # shape (sites, time, d_vars)
        y_s = tf.gather(o_sites, sites)                                  # shape (sites, time, 1)

        site = tf.repeat(tf.range(tf.shape(i)[0]), tf.shape(i)[1])       # shape (queries,)
        i = tf.reshape(i, [-1])                                          # shape (queries,)
        end = i + self.config.lookback + self.config.time_to_target

        query = end - self.config.seq_len + 1                            # encoded step of the query window
        keys = i                                                         # encoded step of the first key
        idx = tf.stack([site, query, keys], axis=-1)                     # shape (queries, 3)

        # Target
        y_t = tf.gather_nd(y_s, tf.stack([site, end - self.config.foresight], axis=-1))[:, 0]
        return (x_s, y_s, sites, idx), y_t

    def _create_dataset(self, f_sites, o_sites) -> tf.data.Dataset:
        """Endless training steps with random sites and random sample indices of the training period."""
        def sample_step(_):
            sites = tf.random.shuffle(tf.range(self.config.max_locations))[:self.sites_per_step]
            i = tf.random.uniform(
                (self.sites_per_step, self.queries_per_site),
                maxval=self.num_train_per_location,
                dtype=tf.int32
            )
            return self._create_step(f_sites, o_sites, sites, i)

        ds = tf.data.Dataset.range(1).repeat()
        return ds.map(sample_step, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    def _create_test_dataset(self, f_sites, o_sites) -> tf.data.Dataset:
        """Every test sample once per epoch: one site per step, consecutive chunks of batch_size samples."""
        chunk_starts = tf.range(self.num_train_per_location, self.num_samples_per_location, self.batch_size)
        ds = tf.data.Dataset.range(self.config.max_locations)
        ds = ds.flat_map(
            lambda loc: tf.data.Dataset.from_tensor_slices(chunk_starts).map(lambda start: (loc, start))
        )

        def extract_step(loc, start):
            sites = tf.cast(tf.reshape(loc, [1]), tf.int32)
            i = tf.range(start, tf.minimum(start + self.batch_size, self.num_samples_per_location))
            return self._create_step(f_sites, o_sites, sites, tf.expand_dims(i, 0))

        return ds.map(extract_step, num_parallel_calls=tf.data.AUTOTUNE).repeat().prefetch(tf.data.AUTOTUNE)

    def get_train_steps(self) -> int:
        """Get the number of training steps per epoch (one pass over the training samples on average)."""
        return int(tf.math.ceil(self.train_size / (self.sites_per_step * self.queries_per_site)))

    def get_test_steps(self) -> int:
        """Get the number of test steps per epoch (every test sample once)."""
        chunks = int(tf.math.ceil(self.num_test_per_location / self.batch_size))
        return chunks * self.config.max_locations
