from a2e.pipeline.base import *

class FastA2EPipeline(BaseTrainingPipeline):
    def __init__(self, *args, **kwargs):
        """
        Specialised TrainingPipeline used to supply a dataset for training FastA2E.
        Every step contains the full series and `batch_size` queries into it.
        Samples are defined as in A2EPipeline.

        Args:
            forecasts: Historical forecast data
            observations: Historical observation data
            config: model configuration
            test_size: Fraction of data to use for testing
        """
        super().__init__(*args, **kwargs)
        self.num_series = self._get_num_series()
        self.sites_per_step = self._get_sites_per_step()
        self.queries_per_site = max(self.batch_size // self.sites_per_step, 1)

        # Samples are split in time per series
        self.num_samples_per_location = int(self.num_timesteps - (self.config.lookback + self.config.time_to_target))
        self.num_test_per_location = int(tf.math.ceil(self.num_samples_per_location * self.test_size_fraction))
        self.num_train_per_location = self.num_samples_per_location - self.num_test_per_location
        self.num_samples = self.num_samples_per_location * self.num_series
        self.test_size = self.num_test_per_location * self.num_series
        self.train_size = self.num_train_per_location * self.num_series

        # Training steps only contain the time steps that training samples use
        self.train_span = self.num_train_per_location + self.config.lookback + self.config.time_to_target

        # Run pipeline setup
        self._validate_inputs()
        self._prepare_dataset()

    def _get_num_series(self) -> int:
        """Number of series in the data"""
        return 1

    def _get_sites_per_step(self) -> int:
        """Number of series encoded per training step"""
        return 1

    def _validate_inputs(self):
        """FastA2E specific assertions"""
        super()._validate_inputs()
        tf.debugging.assert_equal(
            tf.rank(self.forecasts),
            2,
            message="Forecasts must be 2D for FastA2E models"
        )
        tf.debugging.assert_equal(
            tf.rank(self.observations),
            2,
            message="Observations must be 2D for FastA2E models"
        )

    def _to_series(self, f_norm, o_norm):
        """(time, variable) -> (series, time, variable)"""
        return tf.expand_dims(f_norm, 0), tf.expand_dims(o_norm, 0)

    def _inputs(self, x_s, y_s, sites, idx):
        """Model inputs of one step"""
        return x_s, y_s, idx

    def _prepare_dataset(self) -> None:
        """Normalize data and create train and test datasets. Steps are already batched."""
        f_series, o_series = self._to_series(*self._normalize())
        self.train_dataset = self._create_dataset(f_series[:, :self.train_span], o_series[:, :self.train_span])
        self.test_dataset = self._create_test_dataset(f_series, o_series)

    def _create_step(self, f_series, o_series, sites, i):
        """
        Builds one step from the given series and sample indices i (sites, queries_per_site).
        Sample i is the same as in A2EPipeline: lookback f[i: i+lookback], query window ending at
        i+lookback+time_to_target. Indices refer to encoded steps: z[j] <-> window ending at j+seq_len-1.
        """
        x_s = tf.gather(f_series, sites)                                 # shape (sites, time, d_vars)
        y_s = tf.gather(o_series, sites)                                 # shape (sites, time, 1)

        site = tf.repeat(tf.range(tf.shape(i)[0]), tf.shape(i)[1])       # shape (queries,)
        i = tf.reshape(i, [-1])                                          # shape (queries,)
        end = i + self.config.lookback + self.config.time_to_target

        query = end - self.config.seq_len + 1                            # encoded step of the query window
        keys = i                                                         # encoded step of the first key
        idx = tf.stack([site, query, keys], axis=-1)                     # shape (queries, 3)

        # Target
        y_t = tf.gather_nd(y_s, tf.stack([site, end - self.config.foresight], axis=-1))[:, 0]
        return self._inputs(x_s, y_s, sites, idx), y_t

    def _create_dataset(self, f_series, o_series) -> tf.data.Dataset:
        """Endless training steps with random series and random sample indices of the training period."""
        def sample_step(_):
            sites = tf.random.shuffle(tf.range(self.num_series))[:self.sites_per_step]
            i = tf.random.uniform(
                (self.sites_per_step, self.queries_per_site),
                maxval=self.num_train_per_location,
                dtype=tf.int32
            )
            return self._create_step(f_series, o_series, sites, i)

        ds = tf.data.Dataset.range(1).repeat()
        return ds.map(sample_step, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    def _create_test_dataset(self, f_series, o_series) -> tf.data.Dataset:
        """Every test sample once per epoch: one series per step, consecutive chunks of batch_size samples."""
        chunk_starts = tf.range(self.num_train_per_location, self.num_samples_per_location, self.batch_size)
        ds = tf.data.Dataset.range(self.num_series)
        ds = ds.flat_map(
            lambda s: tf.data.Dataset.from_tensor_slices(chunk_starts).map(lambda start: (s, start))
        )

        def extract_step(s, start):
            sites = tf.cast(tf.reshape(s, [1]), tf.int32)
            i = tf.range(start, tf.minimum(start + self.batch_size, self.num_samples_per_location))
            return self._create_step(f_series, o_series, sites, tf.expand_dims(i, 0))

        return ds.map(extract_step, num_parallel_calls=tf.data.AUTOTUNE).repeat().prefetch(tf.data.AUTOTUNE)

    def get_train_steps(self) -> int:
        """Get the number of training steps per epoch (one pass over the training samples on average)."""
        return int(tf.math.ceil(self.train_size / (self.sites_per_step * self.queries_per_site)))

    def get_test_steps(self) -> int:
        """Get the number of test steps per epoch (every test sample once)."""
        chunks = int(tf.math.ceil(self.num_test_per_location / self.batch_size))
        return chunks * self.num_series
