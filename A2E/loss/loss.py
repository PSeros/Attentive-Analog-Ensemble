import tensorflow as tf
from A2E.metrics.computation import compute_crps, compute_scrps

@tf.keras.utils.register_keras_serializable(package="A2E")
class CRPS(tf.keras.losses.Loss):
    def __init__(self, name="crps", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, target, model_output):
        y_h = model_output[:, :, 0]
        w_h = model_output[:, :, 1]
        y_t = target

        if tf.rank(y_t) == 1:
            y_t = tf.expand_dims(y_t, axis=-1)

        loss = -compute_crps(y_h, w_h, y_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="A2E")
class SCRPS(tf.keras.losses.Loss):
    def __init__(self, gamma=1e-8, name="scrps", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, target, model_output):
        y_h = model_output[:, :, 0]
        w_h = model_output[:, :, 1]
        y_t = target

        if tf.rank(y_t) == 1:
            y_t = tf.expand_dims(y_t, axis=-1)

        loss = -compute_scrps(y_h, w_h, y_t, gamma=self.gamma)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
