import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class EpochTracker(Callback):
    """Custom callback to track cumulative epochs"""
    def __init__(self, config, start_epoch, save_path):
        super().__init__()
        self.config = config
        self.start_epoch = start_epoch
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        # Update current_epoch in config (epoch is 0-based within this training session)
        self.config.epochs = self.start_epoch + epoch + 1

        # Save updated config after each epoch
        if self.save_path:
            config_path = os.path.splitext(self.save_path)[0] + "_config.json"
            self.config.save_to_json(config_path)


class BestWeights(Callback):
    """
    Keeps the weights of the epoch with the lowest monitored value (default: val_loss).

    Keras restores the best weights in EarlyStopping only if early stopping triggers. This callback always keeps
    them, so that `restore()` after `fit` sets the weights of the best epoch, also after the full number of epochs.
    The criterion (strictly lower value) is the one of ModelCheckpoint(save_best_only=True).

    `config.best_epoch` (cumulative, 1-based) and `config.best_val_loss` are updated at every improvement, so that the
    config saved by EpochTracker after each epoch matches the checkpoint on disk.

    For resumed training, pass the best value of the previous session; the weights at the start of training
    (the loaded best model) are then the reference until a better epoch is found.
    """
    def __init__(self, config=None, monitor="val_loss", start_epoch=0, best_value=None, best_epoch=None):
        super().__init__()
        self.config = config
        self.monitor = monitor
        self.start_epoch = start_epoch
        self.best = np.inf if best_value is None else float(best_value)
        self.best_epoch = best_epoch
        self.best_weights = None

    def on_train_begin(self, logs=None):
        if np.isfinite(self.best):
            self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        current = float(current)
        if current < self.best:
            self.best = current
            self.best_epoch = self.start_epoch + epoch + 1
            self.best_weights = self.model.get_weights()
            if self.config is not None:
                self.config.best_epoch = self.best_epoch
                self.config.best_val_loss = self.best

    def restore(self):
        """Sets the weights of the best epoch. Returns False if no epoch had a monitored value."""
        if self.best_weights is None:
            return False
        self.model.set_weights(self.best_weights)
        return True


class TrainingMonitor(Callback):
    """
    Adds values to the epoch logs (and thereby to the Keras History):

    - `g`: learned temperature of the cross-attention (only for similarity metrics with g).
    - On a fixed probe batch of training samples, in inference mode, with s the attention logits over the full
      archive, w = softmax(s) and L the training loss:
        - `grad_share_outside_top{k}`: sum_{j not in top-k} |dL/ds_j| / sum_j |dL/ds_j|
        - `weight_share_outside_top{k}`: sum_{j not in top-k} w_j
      Both are means over the probe samples. They show how much of the learning signal training with top-k
      retrieval would discard.

    Runs first in on_epoch_end, before EarlyStopping may restore weights.
    """
    def __init__(self, probe_inputs=None, probe_targets=None, top_k=100):
        super().__init__()
        self.probe_inputs = probe_inputs
        self.probe_targets = probe_targets
        self.top_k = top_k

    @property
    def grad_key(self):
        return f"grad_share_outside_top{self.top_k}"

    @property
    def weight_key(self):
        return f"weight_share_outside_top{self.top_k}"

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        attention = self.model.cross_attention
        if hasattr(attention, "g"):
            logs["g"] = float(attention.g.numpy())
        if self.probe_inputs is not None:
            grad_share, weight_share = self.probe()
            logs[self.grad_key] = grad_share
            logs[self.weight_key] = weight_share

    def probe(self):
        """Gradient and weight share outside the top-k on the probe batch."""
        z_c, z_h, y_h = self.model.encode(self.probe_inputs)
        s = self.model.cross_attention.scores(z_c, z_h)                  # (batch, n_archive)
        with tf.GradientTape() as tape:
            tape.watch(s)
            w = tf.nn.softmax(s)
            output = tf.concat([tf.expand_dims(y_h, -1), tf.expand_dims(w, -1)], axis=-1)
            loss = self.model.loss(self.probe_targets, output)
        grad = tf.abs(tape.gradient(loss, s))
        k = min(self.top_k, int(s.shape[-1]))
        top = tf.math.top_k(s, k=k).indices
        grad_top = tf.reduce_sum(tf.gather(grad, top, batch_dims=1), axis=-1)
        grad_total = tf.reduce_sum(grad, axis=-1)
        grad_share = 1.0 - grad_top / tf.maximum(grad_total, 1e-30)
        weight_share = 1.0 - tf.reduce_sum(tf.gather(w, top, batch_dims=1), axis=-1)
        return float(tf.reduce_mean(grad_share)), float(tf.reduce_mean(weight_share))
