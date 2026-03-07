import os
import tensorflow as tf
from typing import Tuple, List, Union
from A2E.io.config import ModelConfig
from A2E.callbacks.callbacks import EpochTracker
from A2E.io.plotting import plot_training_history
from A2E.factory.factory import Factory

class Api:
    def __init__(self, config: ModelConfig = None):
        self.config = config
        self.factory = Factory(self.config)
        self.model = None
        self.training_history = None

    def train(
            self,
            forecasts: tf.Tensor,
            observations: tf.Tensor,
            optimizer: tf.keras.optimizers.Optimizer,
            loss: tf.keras.losses.Loss,
            epochs: int,
            batch_size: int,
            test_size: float,
            accum_steps: int = 1,
            metrics: List[tf.keras.metrics.Metric] = None,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            save_path: str = None
    ) -> Tuple[tf.keras.models.Model, tf.keras.callbacks.History]:
        """
        Api for Training any A2E model.

        Args:
            forecasts (tf.Tensor): Historical forecast data
            observations (tf.Tensor): Historical observation data
            optimizer (tf.keras.optimizers.Optimizer): Training optimizer
            loss (tf.keras.losses.Loss): Loss function
            epochs (int): Number of epochs to train for
            batch_size (int): Number of samples per batch
            accum_steps (int): Number of batches to accumulate during training before assigning gradients
            test_size (float): Fraction of data for testing
            metrics (tf.keras.metrics.Metric): List of metrics to track
            callbacks (tf.keras.callbacks.Callback): List of callbacks
            save_path (str): Path to save model

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

        # Handle model creation/loading and epoch calculation
        current_epoch = 0
        epochs_to_train = epochs

        if resume_training:
            success, current_epoch = self._load_existing_model(save_path, optimizer, loss, metrics)
            if not success:
                resume_training = False
                current_epoch = 0

        if not resume_training:
            self._create_new_model(optimizer, loss, metrics)

        print(self.model.summary())

        # Calculate target epoch and update config
        target_epoch = current_epoch + epochs_to_train
        self.config.epochs = epochs_to_train  # Store epochs to train in this session

        # Setup callbacks and execute training
        callbacks = self._setup_callbacks(save_path, current_epoch, callbacks)
        self._execute_training(pipeline, epochs_to_train, accum_steps, callbacks)

        # Plot results and save model
        self._plot_and_save_results(save_path, current_epoch, target_epoch, resume_training, epochs_to_train)

        return self.model, self.training_history

    def embed(self, forecasts: tf.Tensor,
              model_path: str = None,
              config_path: str = None,
              verbose: bool = True
              ) -> tf.Tensor:
        """
        Api around the embedding network in trained models, transforming forecast sequences into embedding vectors.

        Args:
            forecasts (tf.Tensor): Forecast sequences to be embedded. Shape (batch, timesteps>=seq_len, features).
            model_path (str): Path to load a trained model from if no model is loaded (optional).
            config_path (str): Path to load a config from if no config is loaded (optional).
            verbose (bool): If False mutes print statements.

        Returns:
            - embeddings (tf.Tensor): Embedded sequences with shape (batch, timesteps-seq_len+1, embedding dimension).
        """
        # Convert to tf.Tensor
        forecasts = tf.cast(forecasts, dtype=tf.float16)
        # Validate input
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
            print(self.model.summary()) if verbose else None
        else:
            assert self.model, "No trained model. Use .train() first or provide save_path."
        if config_path:
            self.config = ModelConfig.load_from_json(config_path)
        assert self.config, "No ModelConfig. Load the Api with a ModelConfig, or provide a config_path."
        assert forecasts.shape[0] >= self.model.seq_len, f"Forecast sequence length is too short. Should be at least {self.model.seq_len}."

        # Normalize data
        forecasts = tf.keras.layers.Normalization(
            mean=self.config.forecast_normalizer_mean,
            variance=self.config.forecast_normalizer_variance
        )(forecasts)

        # Apply the encoder
        forecasts = tf.expand_dims(forecasts, 0) # (time, features) -> (1, time, features)
        embeddings = self.model.encoder(forecasts) # (1, time, features) -> (1, time-seq_len+1, d_model)
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
            self.model = tf.keras.models.load_model(model_path)
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

                # Get current epoch count
                current_epoch = getattr(loaded_config, 'epochs', 0)
                print(f"Loaded config from: {config_path}")
                print(f"model has been trained for {current_epoch} epochs")

            # Load model with custom objects
            self.model = tf.keras.models.load_model(save_path, custom_objects=None)

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
        if callbacks is None:
            callbacks = []

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor="val_loss",
                save_best_only=True,
                mode="min"
            )
            callbacks.append(checkpoint)

            epoch_tracker = EpochTracker(self.config, current_epoch, save_path)
            callbacks.append(epoch_tracker)

        return callbacks

    def _execute_training(self, pipeline, epochs_to_train: int, accum_steps: int,
                          callbacks: List[tf.keras.callbacks.Callback]) -> None:
        """
        Execute the actual model training.

        Args:
            pipeline: Training pipeline
            epochs_to_train: Number of epochs to train
            accum_steps: Number of steps to accumulate gradient
            callbacks: List of callbacks
        """
        if epochs_to_train <= 0:
            print("No training performed.")
            self.training_history = None
            return

        self.training_history = self.model.fit(
            accum_steps,
            pipeline.get_train_data(),
            steps_per_epoch=pipeline.get_train_steps(),
            validation_data=pipeline.get_test_data(),
            validation_steps=pipeline.get_test_steps(),
            epochs=epochs_to_train,
            callbacks=callbacks,
            initial_epoch=0  # Always start from 0 for the current training session
        )

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
        self.model.save(save_path, save_format="keras")

        config_path = os.path.splitext(save_path)[0] + "_config.json"
        self.config.save_to_json(config_path)

        print(f"\nmodel saved at: {os.path.abspath(save_path)}")
        print(f"Config saved at: {os.path.abspath(config_path)}")
        print(f"Total cumulative epochs trained: {self.config.epochs}")
