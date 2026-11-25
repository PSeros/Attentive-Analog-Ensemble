import os
import json
import tensorflow as tf
from typing import Tuple, List, Union
from a2e.io.config import ModelConfig
from a2e.callbacks.callbacks import EpochTracker, BestWeights, TrainingMonitor, TrainingTimer
from a2e.io.plotting import plot_training_history
from a2e.factory.factory import Factory

class Api:
    def __init__(self, config: ModelConfig = None):
        self.config = config
        self.factory = Factory(self.config)
        self.model = None
        self.training_history = None
        self.best_weights = None
        self.monitor = None
        self.timer = None

    def train(
            self,
            forecasts: tf.Tensor,
            observations: tf.Tensor,
            optimizer: tf.keras.optimizers.Optimizer,
            loss: tf.keras.losses.Loss,
            epochs: int,
            batch_size: int,
            test_size: float,
            metrics: List[tf.keras.metrics.Metric] = None,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            save_path: str = None,
            probe_top_k: int = 100,
    ) -> Tuple[tf.keras.models.Model, tf.keras.callbacks.History]:
        """
        Api for Training any A2E model.

        Checkpointing is done by the Api: the model with the lowest val_loss is kept during training
        (ModelCheckpoint at `save_path`), and after training the weights of the best epoch are restored before the
        final save, also if training ran the full number of epochs. `config.best_epoch` and `config.best_val_loss`
        describe the saved weights. Do not pass a ModelCheckpoint in `callbacks`.

        If `config.seed` is set, `tf.keras.utils.set_random_seed(config.seed)` is called before the model is built,
        so that the result of one model does not depend on models trained before it in the same process.
        Op determinism is not enabled (runtime).

        With `config.g_param = "tau"`, τ = 1/g of the cross-attention is excluded from the weight decay of the
        optimizer (AdamW). With "g", g is trained as in the submitted models (with weight decay).

        After the final save, `<model_name>_history.json` is written next to the model (see `_save_history`).
        It is the marker for a finished training run. It contains the training time (TrainingTimer): seconds per
        epoch in `history.epoch_time`, the total in `training_time_s`, and the `device`.

        Args:
            forecasts (tf.Tensor): Historical forecast data
            observations (tf.Tensor): Historical observation data
            optimizer (tf.keras.optimizers.Optimizer): Training optimizer
            loss (tf.keras.losses.Loss): Loss function
            epochs (int): Number of epochs to train for
            batch_size (int): Number of samples per batch
            test_size (float): Fraction of data for testing
            metrics (tf.keras.metrics.Metric): List of metrics to track
            callbacks (tf.keras.callbacks.Callback): List of callbacks
            save_path (str): Path to save model
            probe_top_k (int): k for the gradient and weight share outside the top-k, computed each epoch on a
                fixed probe batch of `batch_size` training samples (TrainingMonitor).

        Returns:
            Tuple of (trained_model (tf.keras.models.Model), training_history (tf.keras.callbacks.History))
        """
        # Convert to tf.Tensor
        forecasts = tf.cast(forecasts, dtype=tf.float32)
        observations = tf.cast(observations, dtype=tf.float32)

        # Check if model already exists and get user decision
        resume_training = False
        if save_path:
            resume_training = self._check_existing_model(save_path)

        # Build the training pipeline
        pipeline = self.factory.build_pipeline(
            forecasts=forecasts,
            observations=observations,
            test_size=test_size,
            batch_size=batch_size
        )

        # Per-epoch monitoring: temperature g, gradient and weight share outside the top-k on a fixed probe batch
        probe_inputs, probe_targets = pipeline.get_probe_data(batch_size)
        self.monitor = TrainingMonitor(probe_inputs, probe_targets, top_k=probe_top_k)

        # Handle model creation/loading and epoch calculation
        current_epoch = 0
        epochs_to_train = epochs

        if resume_training:
            success, current_epoch = self._load_existing_model(save_path, optimizer, loss, metrics)
            if not success:
                resume_training = False
                current_epoch = 0

        if self.config.seed is not None:
            tf.keras.utils.set_random_seed(self.config.seed)

        if not resume_training:
            self._create_new_model(optimizer, loss, metrics)
            self.config.best_epoch = None
            self.config.best_val_loss = None

        self._exclude_g_from_weight_decay()

        print(self.model.summary())

        # Calculate target epoch and update config
        target_epoch = current_epoch + epochs_to_train
        self.config.epochs = epochs_to_train  # Store epochs to train in this session

        # Setup callbacks and execute training
        callbacks = self._setup_callbacks(save_path, current_epoch, callbacks)
        self._execute_training(pipeline, epochs_to_train, callbacks)

        # Plot results and save model
        self._plot_and_save_results(save_path, current_epoch, target_epoch, resume_training, epochs_to_train)

        # History JSON last: marks the training run as finished
        self._save_history(save_path, current_epoch, epochs_to_train, batch_size)

        return self.model, self.training_history

    def embed(self, forecasts: tf.Tensor,
              model_path: str = None,
              config_path: str = None,
              verbose: bool = True,
              location: int = None,
              ) -> tf.Tensor:
        """
        Api around the embedding network in trained models, transforming forecast sequences into embedding vectors.

        Args:
            forecasts (tf.Tensor): Forecast sequences to be embedded. Shape (batch, timesteps>=seq_len, features).
            model_path (str): Path to load a trained model from if no model is loaded (optional).
            config_path (str): Path to load a config from if no config is loaded (optional).
            verbose (bool): If False mutes print statements.
            location (int): Location index for encoders with location embeddings (CA2E, required there). The
                encoder is conditioned on it as in training.

        Returns:
            - embeddings (tf.Tensor): Embedded sequences with shape (batch, timesteps-seq_len+1, embedding dimension).
        """
        # Convert to tf.Tensor
        forecasts = tf.cast(forecasts, dtype=tf.float32)  # same precision as in training
        # Validate input
        if model_path:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(self.model.summary()) if verbose else None
        else:
            assert self.model, "No trained model. Use .train() first or provide save_path."
        if config_path:
            self.config = ModelConfig.load_from_json(config_path)
        assert self.config, "No ModelConfig. Load the Api with a ModelConfig, or provide a config_path."
        assert forecasts.shape[0] >= self.model.seq_len, f"Forecast sequence length is too short. Should be at least {self.model.seq_len}."

        # Normalize data
        forecasts = tf.keras.layers.Normalization(
            mean=self._normalizer_value(self.config.forecast_normalizer_mean),
            variance=self._normalizer_value(self.config.forecast_normalizer_variance)
        )(forecasts)

        # Location conditioning (CA2E): the same location embedding as in training
        loc = None
        if getattr(self.model.encoder, "embedding", None) is not None:
            if location is None:
                raise ValueError(f"{self.model.name} uses location embeddings: pass `location` to embed().")
            loc = tf.constant([location], dtype=tf.int32)
        elif location is not None:
            raise ValueError(f"{self.model.name} has no location embeddings; `location` must be None.")

        # Apply the encoder
        forecasts = tf.expand_dims(forecasts, 0) # (time, features) -> (1, time, features)
        embeddings = self.model.encoder(forecasts, loc=loc) # (1, time, features) -> (1, time-seq_len+1, d_model)
        return tf.squeeze(embeddings, axis=0) # (1, time-seq_len+1, d_model) -> (time-seq_len+1, d_model)

    def retrieve(
            self,
            current_embeddings: tf.Tensor,
            hist_embeddings: tf.Tensor,
            hist_observations: tf.Tensor,
            k: int = False,
            batch_size: int = 32,
            model_path: str = None,
            verbose: bool = True,
    ) -> List[tf.Tensor]:
        """
        Api around the cross-attention layer in trained models using retrieved observations
         and attention weights based on embeddings.

        Args:
            current_embeddings (tf.Tensor): Current embeddings for which the ensembles are to be retrieved.
            hist_embeddings (tf.Tensor): Historical embeddings archive to be searched.
            hist_observations (tf.Tensor): Historical observations archive to be searched.
            k (int): Integer parameter to define the number of ensemble members (optional).
            batch_size (int): Number of samples used per iterative step (Standard: 32).
            model_path (str): Path to load a trained model from if no model is loaded (optional).
            verbose (bool): If False mutes print statements.

        Returns:
            - values (tf.Tensor): Corresponding values.
            - weights (tf.Tensor): Attention Weights for corresponding observations.
        """
        # Convert to tf.Tensor
        current_embeddings = tf.cast(current_embeddings, dtype=tf.float32)
        hist_embeddings = tf.cast(hist_embeddings, dtype=tf.float32)
        hist_observations = tf.cast(hist_observations, dtype=tf.float32)

        # Assert correct shapes
        assert current_embeddings.shape[-1] == hist_embeddings.shape[-1],\
            f"Embedding dimensions must match. But are {current_embeddings.shape[-1]} and {hist_embeddings.shape[-1]}."
        assert hist_embeddings.shape[0] == hist_observations.shape[0],\
            f"Historical length must match. But are {hist_embeddings.shape[0]} and {hist_observations.shape[0]}"
        assert tf.rank(hist_observations).numpy() <= 2,\
            f"Historical observations must be of rank 1 or 2 but is {tf.rank(hist_observations).numpy()}."

        # If given model_path -> load model
        if model_path:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print(self.model.summary()) if verbose else None
        else:
            assert self.model, "No trained model. Use .train() first or provide save_path."

        # Prepare shapes for broadcasting
        current_embeddings = tf.expand_dims(current_embeddings, 1)
        hist_embeddings = tf.expand_dims(hist_embeddings, 0)
        hist_observations = tf.expand_dims(hist_observations, 0)

        # Create batched dataset
        dataset = tf.data.Dataset.from_tensor_slices(current_embeddings).batch(batch_size)

        # Process each batch and collect results
        ensemble = []
        weights = []
        for batch in dataset:
            batch_size_current = tf.shape(batch)[0]
            # Broadcast historical data to match current batch size
            hist_emb_batch = tf.tile(hist_embeddings, (batch_size_current, 1, 1))
            hist_obs_batch = tf.tile(hist_observations, [batch_size_current] + [1] * (tf.rank(hist_observations).numpy() -1))
            batch_ensemble, batch_weights = self.model.cross_attention(batch, hist_emb_batch, hist_obs_batch, k=k)
            ensemble.append(batch_ensemble)
            weights.append(batch_weights)
        # Concatenate all results
        return [tf.concat(ensemble, axis=0), tf.concat(weights, axis=0)]

    @staticmethod
    def _check_existing_model(save_path: str) -> bool:
        """
        Check if model exists and prompt user for resume/overwrite decision.

        Args:
            save_path: Path where model would be saved

        Returns:
            bool: True if model should resume training, False if model should start new
        """
        if not os.path.exists(save_path):
            return False

        config_path = os.path.splitext(save_path)[0] + "_config.json"

        print(f"\nmodel already exists at: {save_path}")
        if os.path.exists(config_path):
            print(f"Config file found at: {config_path}")

        while True:
            choice = input(
                "\nChoose an option:\n"
                "1. Resume training from existing model\n"
                "2. Start new training (overwrite existing model)\n"
                "Enter 1 or 2: "
            ).strip()

            if choice == "1":
                return True
            elif choice == "2":
                print("Will overwrite existing model and start new training.")
                return False
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def _load_existing_model(self, save_path: str, optimizer, loss, metrics) -> Tuple[bool, int]:
        """
        Load existing model and configuration.

        Args:
            save_path: Path to existing model
            optimizer: Training optimizer
            loss: Loss function
            metrics: List of metrics

        Returns:
            Tuple[bool, int]: (success_flag, current_epoch)
        """
        try:
            config_path = os.path.splitext(save_path)[0] + "_config.json"
            current_epoch = 0

            # Load config first to update self.config
            if os.path.exists(config_path):
                loaded_config = ModelConfig.load_from_json(config_path)
                # Update current config with loaded parameters
                self.config.forecast_normalizer_mean = loaded_config.forecast_normalizer_mean
                self.config.forecast_normalizer_variance = loaded_config.forecast_normalizer_variance
                self.config.observation_normalizer_mean = loaded_config.observation_normalizer_mean
                self.config.observation_normalizer_variance = loaded_config.observation_normalizer_variance
                # Best epoch of the previous session (the saved weights)
                self.config.best_epoch = loaded_config.best_epoch
                self.config.best_val_loss = loaded_config.best_val_loss

                # Get current epoch count
                current_epoch = getattr(loaded_config, 'epochs', 0)
                print(f"Loaded config from: {config_path}")
                print(f"model has been trained for {current_epoch} epochs")

            # Load model with custom objects
            self.model = tf.keras.models.load_model(save_path, compile=False)

            # Recompile with new optimizer and loss
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics or [])

            print(f"Resumed model from: {save_path}")
            print("model recompiled with new optimizer and loss function.")

            return True, current_epoch

        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Starting new training instead...")
            return False, 0

    def _create_new_model(self, optimizer, loss, metrics) -> None:
        """
        Create and compile a new model.

        Args:
            optimizer: Training optimizer
            loss: Loss function
            metrics: List of metrics
        """
        self.model = self.factory.build_model(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        print("Created new model.")

    def _exclude_g_from_weight_decay(self) -> None:
        """
        No weight decay (AdamW) on τ = 1/g of the cross-attention (`config.g_param = "tau"`): weight decay is no
        sensible regularization for a temperature, it would pull τ towards 0, i.e. g towards infinity.
        With "g" nothing changes, so that the models train as the submitted ones.
        Must run before the optimizer is built (first gradient step); replaces exclusions that were set on the
        optimizer before.
        """
        attention = self.model.cross_attention
        g_weight = getattr(attention, "g_weight", None)
        optimizer = self.model.optimizer
        if g_weight is None or attention.g_param != "tau" or not hasattr(optimizer, "exclude_from_weight_decay"):
            return
        optimizer.exclude_from_weight_decay(var_list=[g_weight])

    def _setup_callbacks(self, save_path: str, current_epoch: int,
                         callbacks: List[tf.keras.callbacks.Callback]) -> List[tf.keras.callbacks.Callback]:
        """
        Setup training callbacks including checkpointing and epoch tracking.

        Args:
            save_path: Path to save model
            current_epoch: Current epoch number for resumed training
            callbacks: Existing callbacks list

        Returns:
            List of configured callbacks
        """
        callbacks = list(callbacks or [])
        if any(isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks):
            raise ValueError("Checkpointing is done by the Api. Remove the ModelCheckpoint from `callbacks`.")

        # Best weights first: they are taken before EarlyStopping may restore its own best weights.
        # On resume, the best value of the previous session is the reference.
        resumed_best = self.config.best_val_loss if current_epoch > 0 else None
        self.best_weights = BestWeights(
            config=self.config,
            monitor="val_loss",
            start_epoch=current_epoch,
            best_value=resumed_best,
            best_epoch=self.config.best_epoch if current_epoch > 0 else None,
        )
        callbacks.insert(0, self.best_weights)
        if self.monitor is not None:
            callbacks.insert(0, self.monitor)
        # Timer first: epoch_time without the probe of the monitor and without the checkpoint save
        self.timer = TrainingTimer()
        callbacks.insert(0, self.timer)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                initial_value_threshold=resumed_best,
            )
            callbacks.append(checkpoint)

            epoch_tracker = EpochTracker(self.config, current_epoch, save_path)
            callbacks.append(epoch_tracker)

        return callbacks

    def _execute_training(self, pipeline, epochs_to_train: int,
                          callbacks: List[tf.keras.callbacks.Callback]) -> None:
        """
        Execute the actual model training.

        Args:
            pipeline: Training pipeline
            epochs_to_train: Number of epochs to train
            callbacks: List of callbacks
        """
        if epochs_to_train <= 0:
            print("No training performed.")
            self.training_history = None
            return

        self.training_history = self.model.fit(
            pipeline.get_train_data(),
            steps_per_epoch=pipeline.get_train_steps(),
            validation_data=pipeline.get_test_data(),
            validation_steps=pipeline.get_test_steps(),
            epochs=epochs_to_train,
            callbacks=callbacks,
            initial_epoch=0  # Always start from 0 for the current training session
        )

        epoch_times = self.training_history.history.get("epoch_time", [])
        if self.timer.total is not None and epoch_times:
            print(f"Training time: {self.timer.total / 60:.1f} min for {len(epoch_times)} epochs "
                  f"({sum(epoch_times) / len(epoch_times):.1f} s per epoch) on {self._device_name()}")

        # Weights of the best epoch, independent of whether EarlyStopping triggered
        if self.best_weights.restore():
            print(f"Restored weights of the best epoch {self.best_weights.best_epoch} "
                  f"(val_loss {self.best_weights.best:.6f}).")

    def _plot_and_save_results(self, save_path: str, current_epoch: int,
                               target_epoch: int, resume_training: bool, epochs_to_train: int) -> None:
        """
        Plot training history and save model with configuration.

        Args:
            save_path: Path to save model
            current_epoch: Starting epoch number
            target_epoch: Final epoch number
            resume_training: Whether this was resumed training
            epochs_to_train: Number of epochs trained in this session
        """
        if not save_path or epochs_to_train <= 0:
            return

        # Plot training history
        model_name = os.path.basename(save_path).split(".")[0]
        if resume_training:
            suffix = f"_resumed_epoch_{current_epoch}_to_{target_epoch}"
        else:
            suffix = f"_epoch_0_to_{self.config.epochs}"

        plot_training_history(self.training_history, f"{model_name}{suffix}", save_path)

        # Save model and config
        self.model.save(save_path)

        config_path = os.path.splitext(save_path)[0] + "_config.json"
        self.config.save_to_json(config_path)

        print(f"\nmodel saved at: {os.path.abspath(save_path)}")
        print(f"Config saved at: {os.path.abspath(config_path)}")
        print(f"Total cumulative epochs trained: {self.config.epochs}")
        print(f"Saved weights: epoch {self.config.best_epoch} (val_loss {self.config.best_val_loss})")

    @staticmethod
    def _normalizer_value(value):
        if value is None:
            return None
        tensor = tf.constant(value, dtype=tf.float32)
        # Stats adapted over (time, ..., features) keep a 1 per reduced axis: A2E (1, F), CA2E/SA2E (1, 1, F).
        # Keras 3 Normalization(axis=-1) needs shape (F,).
        while tensor.shape.rank and tensor.shape.rank > 1 and tensor.shape[0] == 1:
            tensor = tf.squeeze(tensor, axis=0)
        return tensor.numpy().tolist()

    @staticmethod
    def _device_name() -> str:
        """Name of the GPU(s) used for training, or "CPU"."""
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return "CPU"
        names = [tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name) for gpu in gpus]
        return ", ".join(names)

    @staticmethod
    def history_path(save_path: str) -> str:
        """Path of the history JSON of a model (`<model_name>_history.json` next to the model)."""
        return os.path.splitext(save_path)[0] + "_history.json"

    def _save_history(self, save_path: str, current_epoch: int, epochs_to_train: int, probe_size: int) -> None:
        """
        Writes `<model_name>_history.json`:
            seed, config, initial_epoch, epochs_trained, best_epoch, best_val_loss, stopped_early,
            training_time_s, device, probe (size, top_k) and history: one list per Keras log (loss, crps, entropy,
            ess, lr, ... and the val_ variants), g and the gradient/weight share outside the top-k (TrainingMonitor),
            epoch_time (TrainingTimer).
        Epoch numbers are cumulative and 1-based; `history` and `training_time_s` cover this session.
        Training times are wall-clock seconds (see TrainingTimer); they depend on the device and on other jobs
        running on it.
        """
        if not save_path or self.training_history is None:
            return

        def to_json(value):
            return None if value is None else float(value)

        history = {key: [to_json(v) for v in values] for key, values in self.training_history.history.items()}
        n_epochs = len(history.get("loss", []))
        for key in ["g", self.monitor.grad_key, self.monitor.weight_key, "epoch_time"]:
            history.setdefault(key, [None] * n_epochs)

        record = {
            "seed": self.config.seed,
            "config": self.config.to_dict(),
            "initial_epoch": current_epoch,
            "epochs_trained": self.config.epochs,
            "best_epoch": self.config.best_epoch,
            "best_val_loss": self.config.best_val_loss,
            "stopped_early": n_epochs < epochs_to_train,
            "training_time_s": to_json(self.timer.total),
            "device": self._device_name(),
            "probe": {"size": probe_size, "top_k": self.monitor.top_k,
                      "samples": "evenly spaced over the training samples, inference mode"},
            "history": history,
        }
        path = self.history_path(save_path)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"History saved at: {os.path.abspath(path)}")
