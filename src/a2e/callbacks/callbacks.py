import os
import numpy as np
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
