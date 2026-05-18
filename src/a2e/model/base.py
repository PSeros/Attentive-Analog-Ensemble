import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="A2E")
class BaseModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Retained for compatibility with models saved before Keras-native
        # optimizer gradient accumulation was used.
        self.accum_counter = self.add_weight(
            name="accum_counter",
            shape=(),
            dtype=tf.int32,
            initializer="zeros",
            trainable=False
        )

    def get_config(self):
        """Extract base model configuration."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Create BaseModel instance from config."""
        return cls(**config)
