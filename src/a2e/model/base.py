import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="A2E")
class BaseModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gradient accumulation is a feature of Keras 3.
        # Due to this package operates on Keras 2, we implemented it in this class.
        self.accum_steps = None

        # Gradient Accumulation
        self._grad_accum_dict = {}  # Dictionary for gradient accumulators
        self._current_var_signatures = []  # Signatures of current variables

        self.accum_counter = self.add_weight(
            name="accum_counter",
            shape=(),
            dtype=tf.int32,
            initializer="zeros",
            trainable=False
        )

    def get_config(self):
        """Extract base model configuration"""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Create BaseModel instance from config"""
        return cls(**config)

    def fit(self, accum_steps, *args, **kwargs):
        """Trains the model for a fixed number of epochs (dataset iterations).

        Args:
            x: Input data. It could be:
              - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
              - A `tf.data` dataset. Should return a tuple
                of either `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
              - A generator or `keras.utils.Sequence` returning `(inputs,
                targets)` or `(inputs, targets, sample_weights)`.
              - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
                callable that takes a single argument of type
                `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
                `DatasetCreator` should be used when users prefer to specify the
                per-replica batching and sharding logic for the `Dataset`.
                See `tf.keras.utils.experimental.DatasetCreator` doc for more
                information.
              A more detailed description of unpacking behavior for iterator
              types (Dataset, generator, Sequence) is given below. If these
              include `sample_weights` as a third component, note that sample
              weighting applies to the `weighted_metrics` argument but not the
              `metrics` argument in `compile()`. If using
              `tf.distribute.experimental.ParameterServerStrategy`, only
              `DatasetCreator` type is supported for `x`.
            y: Target data. Like the input data `x`,
              it could be either Numpy array(s) or TensorFlow tensor(s).
              It should be consistent with `x` (you cannot have Numpy inputs and
              tensor targets, or inversely). If `x` is a dataset, generator,
              or `keras.utils.Sequence` instance, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of datasets, generators, or `keras.utils.Sequence`
                instances (since they generate batches).
            accum_steps: Integer or `None`.
                Number of batches to accumulate before applying the gradient.
                If unspecified, `accum_steps` will default to 1 (i.e. no accumulation).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided
                (unless the `steps_per_epoch` flag is set to
                something other than None).
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 'auto', 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                'auto' becomes 1 for most cases, but 2 when used with
                `ParameterServerStrategy`. Note that the progress bar is not
                particularly useful when logged to a file, so verbose=2 is
                recommended when not running interactively (eg, in a production
                environment). Defaults to 'auto'.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`. Note
                `tf.keras.callbacks.ProgbarLogger` and
                `tf.keras.callbacks.History` callbacks are created automatically
                and need not be passed into `model.fit`.
                `tf.keras.callbacks.ProgbarLogger` is created or not based on
                `verbose` argument to `model.fit`.
                Callbacks with batch-level calls are currently unsupported with
                `tf.distribute.experimental.ParameterServerStrategy`, and users
                are advised to implement epoch-level calls instead with an
                appropriate `steps_per_epoch` value.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This
                argument is not supported when `x` is a dataset, generator or
                `keras.utils.Sequence` instance.
                If both `validation_data` and `validation_split` are provided,
                `validation_data` will override `validation_split`.
                `validation_split` is not yet supported with
                `tf.distribute.experimental.ParameterServerStrategy`.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data. Thus, note the fact
                that the validation loss of data provided using
                `validation_split` or `validation_data` is not affected by
                regularization layers like noise and dropout.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
                  - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
                    arrays.
                  - A `tf.data.Dataset`.
                  - A Python generator or `keras.utils.Sequence` returning
                  `(inputs, targets)` or `(inputs, targets, sample_weights)`.
                `validation_data` is not yet supported with
                `tf.distribute.experimental.ParameterServerStrategy`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch'). This argument is
                ignored when `x` is a generator or an object of tf.data.Dataset.
                'batch' is a special option for dealing
                with the limitations of HDF5 data; it shuffles in batch-sized
                chunks. Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must be
                one-hot encoded, or an explicit final dimension of `1` must
                be included for sparse class labels.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                This argument is not supported when `x` is a dataset, generator,
                or `keras.utils.Sequence` instance, instead provide the
                sample_weights as the third element of `x`.
                Note that sample weighting does not apply to metrics specified
                via the `metrics` argument in `compile()`. To apply sample
                weighting to your metrics, you can specify them via the
                `weighted_metrics` in `compile()` instead.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If x is a
                `tf.data` dataset, and 'steps_per_epoch'
                is None, the epoch will run until the input dataset is
                exhausted.  When passing an infinitely repeating dataset, you
                must specify the `steps_per_epoch` argument. If
                `steps_per_epoch=-1` the training will run indefinitely with an
                infinitely repeating dataset.  This argument is not supported
                with array inputs.
                When using `tf.distribute.experimental.ParameterServerStrategy`:
                  * `steps_per_epoch=None` is not supported.
            validation_steps: Only relevant if `validation_data` is provided and
                is a `tf.data` dataset. Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If 'validation_steps' is None,
                validation will run until the `validation_data` dataset is
                exhausted. In the case of an infinitely repeated dataset, it
                will run into an infinite loop. If 'validation_steps' is
                specified and only part of the dataset will be consumed, the
                evaluation will start from the beginning of the dataset at each
                epoch. This ensures that the same validation samples are used
                every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is in
                the form of datasets, generators, or `keras.utils.Sequence`
                instances (since they generate batches).
            validation_freq: Only relevant if validation data is provided.
              Integer or `collections.abc.Container` instance (e.g. list, tuple,
              etc.).  If an integer, specifies how many training epochs to run
              before a new validation run is performed, e.g. `validation_freq=2`
              runs validation every 2 epochs. If a Container, specifies the
              epochs on which to run validation, e.g.
              `validation_freq=[1, 2, 10]` runs validation at the end of the
              1st, 2nd, and 10th epochs.
            max_queue_size: Integer. Used for generator or
              `keras.utils.Sequence` input only. Maximum size for the generator
              queue.  If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-pickleable arguments to
                the generator as they can't be passed easily to children
                processes.

        Unpacking behavior for iterator-like inputs:
            A common pattern is to pass a tf.data.Dataset, generator, or
          tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
          yield not only features (x) but optionally targets (y) and sample
          weights.  Keras requires that the output of such iterator-likes be
          unambiguous. The iterator should return a tuple of length 1, 2, or 3,
          where the optional second and third elements will be used for y and
          sample_weight respectively. Any other type provided will be wrapped in
          a length one tuple, effectively treating everything as 'x'. When
          yielding dicts, they should still adhere to the top-level tuple
          structure.
          e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
          features, targets, and weights from the keys of a single dict.
            A notable unsupported data type is the namedtuple. The reason is
          that it behaves like both an ordered datatype (tuple) and a mapping
          datatype (dict). So given a namedtuple of the form:
              `namedtuple("example_tuple", ["y", "x"])`
          it is ambiguous whether to reverse the order of the elements when
          interpreting the value. Even worse is a tuple of the form:
              `namedtuple("other_tuple", ["x", "y", "z"])`
          where it is unclear if the tuple was intended to be unpacked into x,
          y, and sample_weight or passed through as a single element to `x`. As
          a result the data processing code will simply raise a ValueError if it
          encounters a namedtuple. (Along with instructions to remedy the
          issue.)

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: 1. If the model was never compiled or,
            2. If `model.fit` is  wrapped in `tf.function`.

            ValueError: In case of mismatch between the provided input data
                and what the model expects or when the input data is empty.
        """
        self.accum_steps = tf.constant(accum_steps, dtype=tf.int32)
        return super().fit(*args, **kwargs)

    def train_step(self, data):
        """Train Step with Gradient Accumulation. (Apply Gradient after several epochs)"""
        input, target = data

        # Ensure gradient accumulator is initialized
        self._ensure_grad_accum_initialized()

        with tf.GradientTape() as tape:
            output = self(input, training=True)
            loss = self.compute_loss(y_pred=output, y=target)
            scaled_loss = loss / tf.cast(self.accum_steps, loss.dtype)

        gradients = tape.gradient(scaled_loss, self.trainable_variables)

        # Accumulate Gradient
        for var, grad in zip(self.trainable_variables, gradients):
            signature = self._get_var_signature(var)
            if signature in self._grad_accum_dict:
                self._grad_accum_dict[signature].assign_add(grad)

        # Increment Counter
        self.accum_counter.assign_add(1)

        # Check for accumulation end
        tf.cond(
            tf.equal(self.accum_counter, self.accum_steps),
            true_fn=lambda: self._apply_and_reset_gradients(),
            false_fn=lambda: tf.no_op()
        )

        return self.compute_metrics(y_pred=output, y=target, x=input, sample_weight=None)

    def test_step(self, data):
        input, target = data
        output = self(input, training=False)
        self.compute_loss(y_pred=output, y=target)
        return self.compute_metrics(y_pred=output, y=target, x=input, sample_weight=None)

    def reset_gradient_accumulator(self):
        """Public method for manually resetting the gradient accumulator"""
        self._current_var_signatures = []
        self._grad_accum_dict.clear()
        self.accum_counter.assign(0)

    def _apply_and_reset_gradients(self):
        """Apply accumulated Gradients and reset accumulator"""
        # Create lists for apply_gradients
        accumulated_grads = []
        trainable_vars = []

        for var in self.trainable_variables:
            signature = self._get_var_signature(var)
            if signature in self._grad_accum_dict:
                accumulated_grads.append(self._grad_accum_dict[signature])
                trainable_vars.append(var)

        # Apply accumulated Gradients
        if accumulated_grads and trainable_vars:
            self.optimizer.apply_gradients(zip(accumulated_grads, trainable_vars))

        # Reset Accumulator
        for accum_var in self._grad_accum_dict.values():
            accum_var.assign(tf.zeros_like(accum_var))

        self.accum_counter.assign(0)
        return tf.no_op()

    def _get_var_signature(self, var):
        """Creates a unique signature for a variable"""
        return f"{var.name}_{var.shape}_{var.dtype.name}"

    def _ensure_grad_accum_initialized(self):
        """Ensures that the gradient accumulator is initialized for current trainable_variables"""
        current_signatures = [self._get_var_signature(var) for var in self.trainable_variables]

        # Check if reinitialization is needed
        if current_signatures != self._current_var_signatures:
            # Clear old accumulator variables
            self._grad_accum_dict.clear()

            # Create new accumulator variables
            for var in self.trainable_variables:
                signature = self._get_var_signature(var)
                # Use tf.Variable directly instead of add_weight
                self._grad_accum_dict[signature] = tf.Variable(
                    tf.zeros_like(var),
                    trainable=False,
                    name=f"grad_accum_{len(self._grad_accum_dict)}"
                )

            self._current_var_signatures = current_signatures
            # Reset counter on reinitialization
            self.accum_counter.assign(0)
