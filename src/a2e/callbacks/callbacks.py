import os
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