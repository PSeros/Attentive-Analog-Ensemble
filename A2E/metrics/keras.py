import tensorflow as tf
from A2E.metrics.computation import compute_crps, compute_scrps, compute_entropy, compute_cross_entropy

@tf.keras.utils.register_keras_serializable(package="A2E")
class CRPSMetric(tf.keras.metrics.Mean):
    def __init__(self, name="crps", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, target, model_output, sample_weight=None):
        x = model_output[:, :, 0]
        w = model_output[:, :, 1]
        y = target

        if tf.rank(y) == 1:
            y = tf.expand_dims(y, axis=-1)

        value = -compute_crps(x, w, y)
        super().update_state(value)

@tf.keras.utils.register_keras_serializable(package="A2E")
class SCRPSMetric(tf.keras.metrics.Mean):
    def __init__(self, gamma=1e-8, name="scrps", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
    def update_state(self, target, model_output, sample_weight=None):
        x = model_output[:, :, 0]
        w = model_output[:, :, 1]
        y = target

        if tf.rank(y) == 1:
            y = tf.expand_dims(y, axis=-1)

        value = -compute_scrps(x, w, y, gamma=self.gamma)
        super().update_state(value)

@tf.keras.utils.register_keras_serializable(package="A2E")
class EntropyMetric(tf.keras.metrics.Mean):
    def __init__(self, beta=0.01, eps=1e-8, name="entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.eps = eps

    def update_state(self, target, model_output, sample_weight=None):
        W = model_output[:, :, 1]
        penalty = self.beta * compute_entropy(W, self.eps)
        super().update_state(penalty)

@tf.keras.utils.register_keras_serializable(package="A2E")
class CrossEntropyMetric(tf.keras.metrics.Mean):
    def __init__(self, beta=0.01, eps=1e-8, name="cross-entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.eps = eps

    def update_state(self, target, model_output, sample_weight=None):
        W = model_output[:, :, 1]
        penalty = self.beta * compute_cross_entropy(W, self.eps)
        super().update_state(penalty)