import tensorflow as tf
from abc import ABC, abstractmethod
from A2E.io.config import ModelConfig

class BaseTrainingPipeline(ABC):
    def __init__(
            self,
            forecasts: tf.Tensor,
            observations: tf.Tensor,
            config: ModelConfig,
            test_size: float,
            batch_size: int,):
        """
        Base class for TrainingPipelines used to supply a dataset for training.

        Args:
            forecasts (np.ndarray): Historical forecast data.
            observations (np.ndarray): Historical observation data.
            config (ModelConfig): model configuration.
            test_size (float): Fraction of data to use for testing.
        """
        self.forecasts = tf.cast(forecasts, dtype=tf.float32)
        self.observations = tf.cast(observations, dtype=tf.float32)
        self.test_size_fraction = test_size
        self.num_timesteps = self.forecasts.shape[0]
        self.config = config
        self.batch_size = batch_size
        self.train_size = None
        self.test_size = None

        # Initialize datasets
        self.train_dataset = None
        self.test_dataset = None
        self.full_dataset = None

        # Initialize Normalizer
        self.f_normalizer = tf.keras.layers.Normalization()
        self.o_normalizer = tf.keras.layers.Normalization()

    def _prepare_dataset(self) -> None:
        """Create and adapt normalizers on training data."""
        time_steps = self.forecasts.shape[0]
        idx = int(time_steps - tf.math.ceil(time_steps * self.test_size_fraction))
        f_train, f_test = tf.split(self.forecasts, [idx, -1])
        o_train, o_test = tf.split(self.observations, [idx, -1])

        # Adapt Normalizer on training data
        self.f_normalizer.adapt(f_train)
        self.o_normalizer.adapt(o_train)

        # Normalize full dataset
        full_f_norm = self.f_normalizer(self.forecasts)
        full_o_norm = self.o_normalizer(self.observations)

        # Store normalization parameters in config
        self.config.forecast_normalizer_mean = self.f_normalizer.mean.numpy().tolist()
        self.config.forecast_normalizer_variance = self.f_normalizer.variance.numpy().tolist()
        self.config.observation_normalizer_mean = self.o_normalizer.mean.numpy().tolist()
        self.config.observation_normalizer_variance = self.o_normalizer.variance.numpy().tolist()

        # Create sequential dataset
        self.full_dataset = self._create_dataset(full_f_norm, full_o_norm)

        # Split, batch and prefetch datasets
        train_dataset = self.full_dataset.take(self.train_size).repeat()
        self.train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        test_dataset = self.full_dataset.skip(self.train_size).repeat()
        self.test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def get_train_data(self) -> tf.data.Dataset:
        """Get training dataset."""
        return self.train_dataset

    def get_test_data(self) -> tf.data.Dataset:
        """Get test dataset."""
        return self.test_dataset

    def get_train_steps(self) -> int:
        """Get the number of training steps per epoch."""
        return int(tf.math.ceil(self.train_size / self.batch_size))

    def get_test_steps(self) -> int:
        """Get the number of test steps per epoch."""
        return int(tf.math.ceil(self.test_size / self.batch_size))

    @abstractmethod
    def _validate_inputs(self):
        # General assertions
        tf.debugging.assert_all_finite(self.forecasts, message="NaN/Inf in forecasts")
        tf.debugging.assert_all_finite(self.observations, message="NaN/Inf in observations")
        tf.debugging.assert_equal(
            tf.shape(self.forecasts)[0],
            tf.shape(self.observations)[0],
            message="Forecasts and observations must have same length"
        )
        tf.debugging.assert_equal(
            tf.shape(self.forecasts)[-1],
            self.config.d_vars,
            message="Last dimension of forecasts must match number of Features"
        )
        tf.debugging.assert_equal(
            tf.shape(self.observations)[-1],
            1,
            message="Last dimension of observations must be 1"
        )

    @abstractmethod
    def _create_dataset(self, f_data, o_data):
        pass
