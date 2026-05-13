import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="A2E")
class PaddingLayer(tf.keras.layers.Layer):
    def __init__(self, dilation_rate, locations_kernel_size):
        super().__init__()
        self.dilation_rate = dilation_rate
        self.locations_kernel_size = locations_kernel_size
        assert locations_kernel_size % 2 == 1, "locations_kernel_size must be odd."
    def call(self, x):
        """
        Manual padding to ensure causality in time dimension.
        No causality needed for spatial dimension.
        """
        # Causal padding for time dimension; symmetrical für spatial dimension
        locations_pad = (self.locations_kernel_size - 1) // 2
        x_padded = tf.pad(x, [[0, 0], [self.dilation_rate, 0], [locations_pad, locations_pad], [0, 0]])
        return x_padded

    def get_config(self):
        config = super().get_config()
        config.update({
            "dilation_rate": self.dilation_rate,
            "locations_kernel_size": self.locations_kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="A2E")
class SpatialDilatedConvLayer(tf.keras.layers.Layer):
    """
    Spatial Dilated Convolutional layer for 4D tensors (batch, time, locations, d_model)
    Combines temporal dilated convolution with spatial convolution for spatio-temporal data.
    Supports conditional processing similar to the conditional WaveNet.
    """
    def __init__(self, filter, seq_len, locations_kernel_size=3, dropout=.0, name=None, **kwargs):
        """
        Initializes a dilated convolutional layer for spatiotemporal data used in SpatialWaveNet model.
        Using Gated Activation Units.

        Args:
            filter (int): The number of filters (output channels) for convolutional layers.
            seq_len (int): Length of the input sequence.
            dropout (float): Dropout rate for regularization.
            name (str, optional): Name of the layer. Defaults to None.
            **kwargs: Additional keyword arguments for parent class initialization.
        """
        super().__init__(name=name, **kwargs)
        self.filter = filter
        self.seq_len = seq_len
        self.locations_kernel_size = locations_kernel_size
        self.dropout = dropout

        # Calculate dilated convolution layers
        kernel_size = 2
        _seq_len = tf.constant(self.seq_len, dtype=tf.float32)
        _kernel_size = tf.constant(kernel_size, dtype=tf.float32)
        _num_layers = tf.math.log(_seq_len) / tf.math.log(_kernel_size)
        num_layers = int(tf.math.ceil(_num_layers))

        # Gated dilated convolution components (f and g)
        # Use Conv2D with dilation along time dimension only
        self.W_f = tf.keras.Sequential([
            layer
            for i in range(num_layers)
                for layer in
                    [
                        PaddingLayer(kernel_size ** i, locations_kernel_size),
                        tf.keras.layers.Conv2D(
                            filters=filter,
                            kernel_size=(kernel_size, locations_kernel_size),  # (time, locations)
                            strides=(1, 1),
                            dilation_rate=(kernel_size ** i, 1),  # dilate only along time
                            padding='valid',   # no padding here; padding is added externally
                            use_bias=False
                        )
                    ]
        ])

        self.W_g = tf.keras.Sequential([
            layer
            for i in range(num_layers)
                for layer in
                    [
                        PaddingLayer(kernel_size ** i, locations_kernel_size),
                        tf.keras.layers.Conv2D(
                            filters=filter,
                            kernel_size=(kernel_size, locations_kernel_size),  # (time, locations)
                            strides=(1, 1),
                            dilation_rate=(kernel_size ** i, 1),  # dilate only along time
                            padding='valid',  # no padding here; padding is added externally
                            use_bias=False
                        )
                    ]
        ])

        # Conditional Bias V|h (only used when h is provided)
        self.V_f = tf.keras.layers.Dense(filter, use_bias=False)
        self.V_g = tf.keras.layers.Dense(filter, use_bias=False)

        # Regularization - adapted for 4D tensors
        self.dropout = tf.keras.layers.Dropout(
            rate=dropout,
            noise_shape=(None, 1, 1, filter)  # For 4D: (batch, time, locations, features)
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            x = input_shape[0]
        else:
            x = input_shape
        return x

    def call(self, x, h=None, training=None):
        """
        Applies the dilated convolution and conditional operations to the input tensor.

        Args:
            x (tensor): Input tensor for the layer with shape(batch_size, time, location, d_model).
            h (tensor, optional): Conditional input tensor with shape(batch_size, location, d_model). Defaults to None.
            training (bool, optional): Whether the model is in training mode or not. Defaults to None.

        Returns:
            tensor: The output tensor after applying dilated convolutions, conditional biases,
                   gated activation, and dropout, with shape (batch_size, time, location, d_model).
        """
        # Apply gated convolutions
        Wx_f = self.W_f(x)
        Wx_g = self.W_g(x)

        if h is not None:
            # Conditional case: add conditioning signal
            # h is of shape(batch, locations, d_model) for location-specific conditioning
            Vh_f = self.V_f(h)
            Vh_g = self.V_g(h)

            if len(h.shape) == 2:
                # Global conditioning: (batch, d_model) -> (batch, 1, 1, d_model)
                Vh_f = tf.expand_dims(tf.expand_dims(Vh_f, 1), 1)
                Vh_g = tf.expand_dims(tf.expand_dims(Vh_g, 1), 1)
            else:
                # Location-specific conditioning: (batch, locations, d_model) -> (batch, 1, locations, d_model)
                Vh_f = tf.expand_dims(Vh_f, 1)
                Vh_g = tf.expand_dims(Vh_g, 1)

            f = tf.add(Wx_f, Vh_f)
            g = tf.add(Wx_g, Vh_g)
        else:
            # Non-conditional case: use only convolution outputs
            f = Wx_f
            g = Wx_g

        # Gated Activation
        out = tf.nn.tanh(f) * tf.nn.sigmoid(g)
        out = self.dropout(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filter": self.filter,
            "seq_len": self.seq_len,
            "locations_kernel_size": self.locations_kernel_size,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="A2E")
class SpatialWaveNetBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, locations_kernel_size=3, dropout=.0, name=None, **kwargs):
        """
        Initializes a block used in the WaveNet architecture.
        Consisting of a Dilated Convolution and a linear projection.
        Including Residual and Scip connections.

        Args:
            d_model (int): Dimensionality of the model, corresponding to the number of filters.
            seq_len (int): Length of the input sequence.
                        locations_kernel_size (int, optional): Kernel size of the convolutional kernel for the spatial dimension. Defaults to 3.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.
            name (str, optional): Name of the layer. Defaults to None.
            **kwargs: Additional keyword arguments for parent class initialization.
        """
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.locations_kernel_size = locations_kernel_size
        self.dropout = dropout

        self.dilated_conv = SpatialDilatedConvLayer(
            filter=d_model,
            seq_len=seq_len,
            locations_kernel_size=locations_kernel_size,
            dropout=dropout,
        )
        self.ff = tf.keras.layers.Dense(d_model)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            x = input_shape[0]
        else:
            x = input_shape
        return (x, x)

    def call(self, residual, h=None, skip=None, training=None):
        """
        Applies the SpatialWaveNet block operations to the residual tensor and updates skip connections.

        Args:
            residual (tensor): The input residual tensor with shape(batch_size, time, d_model).
            h (tensor, optional): Conditional input tensor with shape(batch_size, d_model). Defaults to None.
            skip (tensor, optional): The skip connection tensor from previous blocks with shape(batch_size, seq_len, d_model). Defaults to 0.
            training (bool, optional): Whether the model is in training mode or not. Defaults to None.

        Returns:
            tuple: A tuple containing the updated residual tensor with shape(batch_size, time, locations, d_model)
                  and the updated skip connection tensor with shape(batch_size, time, locations, d_model).
        """
        dilated_conv_output = self.dilated_conv(residual, h, training=training)
        ff_output = self.ff(dilated_conv_output)

        if skip is not None:
            skip = tf.add(skip, ff_output)
        else:
            skip = ff_output

        residual = tf.add(residual, ff_output)
        return residual, skip

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "seq_len": self.seq_len,
            "locations_kernel_size": self.locations_kernel_size,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="A2E")
class SpatialWaveNet(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, d_loc, n_blocks, locations_kernel_size=3, dropout=.0, name="SpatialWaveNet", **kwargs):
        """
        Initializes a WaveNet model with multiple WaveNet blocks adapted for the spatial dimension.
        Including two final Dense Layers (first with ReLu and second Linear) which output the final embedding z.

        Args:
            d_model (int): Dimensionality of the model.
            seq_len (int): Length of the input sequence.
            n_blocks (int): Number of blocks in the WaveNet.
            locations_kernel_size (int, optional): Kernel size of the convolutional kernel for the spatial dimension. Defaults to 3.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.
            name (str, optional): Name of the layer. Defaults to "SpatialWaveNet".
            **kwargs: Additional keyword arguments for parent class initialization.
        """
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.d_loc = d_loc
        self.n_blocks = n_blocks
        self.locations_kernel_size = locations_kernel_size
        self.dropout_rate = dropout

        # Key Components
        self.wave_net_blocks = [SpatialWaveNetBlock(
            d_model=d_model,
            seq_len=seq_len // n_blocks,
            locations_kernel_size=locations_kernel_size,
            dropout=dropout,
        ) for _ in range(n_blocks)]

        self.dense1 = tf.keras.layers.Dense(d_model, activation="relu")
        self.dense2 = tf.keras.layers.Dense(d_model, activation=None)

        # Regularization - for 3D tensors
        self.dropout = tf.keras.layers.Dropout(
            rate=self.dropout_rate,
            noise_shape=(None, 1, self.d_model)
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        else:
            return input_shape

    def call(self, x, h=None, training=None):
        """
        Applies multiple SpatialWaveNet blocks to the input tensor and processes the result.

        Args:
            x (tensor): Input tensor for the model with shape(batch_size, time, d_model).
            h (tensor, optional): Conditional input tensor with shape(batch_size, d_model). Defaults to None.
            training (bool, optional): Whether the model is in training mode or not. Defaults to None.

        Returns:
            tensor: The output tensor after processing through SpatialWaveNet blocks, dense layers, and dropout,
                   with shape(batch_size, time-seq_len+1, d_model).
        """
        residual = x
        skip = None

        for block in self.wave_net_blocks:
            residual, skip = block(residual, h, skip, training=training)

        # Cropping padded temporal regions
        skip = skip[:, self.seq_len-1:, :, :]

        # Final transformations
        skip = tf.nn.relu(skip)
        pool = tf.reduce_mean(skip, axis=2, keepdims=False)
        pool = self.dense1(pool)
        pool = self.dropout(pool, training=training)
        z = self.dense2(pool)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "seq_len": self.seq_len,
            "d_loc": self.d_loc,
            "n_blocks": self.n_blocks,
            "locations_kernel_size": self.locations_kernel_size,
            "dropout": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="A2E")
class SpatialEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, d_loc, seq_len, n_blocks, dropout=.0, max_locations=None,
                 locations_kernel_size=3, name="SpatialEncoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.d_loc = d_loc
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.max_locations = max_locations
        self.locations_kernel_size = locations_kernel_size

        # Projection
        self.projection = tf.keras.layers.Conv2D(
            filters=d_model,
            kernel_size=(1, d_loc),
            padding="same",
            name="projection"
        )

        # Embedding (for Conditional Encoder)
        self.embedding = None
        if max_locations:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=max_locations,
                output_dim=d_model,
                name="loc_embedding"
            )

        # Spatial WaveNet
        self.spatial_wave_net = SpatialWaveNet(
            d_model=d_model,
            seq_len=seq_len,
            d_loc=d_loc,
            n_blocks=n_blocks,
            locations_kernel_size=locations_kernel_size,
            dropout=dropout,
            name="spatial_wavenet"
        )

    def build(self, input_shape):
        """Expected input shape: (batch, time, locations, features)"""
        super().build(input_shape)
        # Expected input shape: (batch, time, locations, features)
        _, _, d_loc, d_vars = input_shape

        # Build layers
        self.projection.build((None, None, self.d_loc, d_vars))
        self.spatial_wave_net.build([(None, None, self.d_loc, self.d_model), (None, self.d_model)])

        if self.embedding:
            self.embedding.build((None,))

    def compute_output_shape(self, input_shape):
        _, _, d_loc, d_vars = input_shape
        return (None, self.d_loc, self.d_model)

    def call(self, x, loc=None, training=None):
        """
        Projects the input tensor and encodes it using location embeddings and a SpatialWaveNet model.

        Args:
            x (tensor): Input tensor for the encoder with shape(batch_size, time, locations, d_vars).
            loc (tensor, optional): Location input tensor for conditional embedding with shape(batch_size, locations). Defaults to None.
            training (bool, optional): Flag indicating whether the layer is in training mode.

        Returns:
            tensor: The encoded output tensor with shape(batch_size, time-seq_len+1, d_model).
            The Spatial dim is reduced to the target location and removed.
        """
        # Project to d_model
        x = self.projection(x)

        # Get location embedding if used
        h = None
        if loc is not None and self.embedding is not None:
            h = self.embedding(loc)

        # Apply SpatialWaveNet encoding
        z = self.spatial_wave_net(x, h, training=training)
        return z

    def get_config(self):
        """Extract configuration"""
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'd_loc': self.d_loc,
            'n_blocks': self.n_blocks,
            'locations_kernel_size': self.locations_kernel_size,
            'max_locations': self.max_locations,
            'dropout': self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create instance from config"""
        return cls(**config)

# Execution Test
if __name__ == "__main__":
    # Parameters for the encoder
    d_model = 8  # Dimension of the model
    seq_len = 1*(2**4)  # Length of the input sequence
    n_blocks = 1  # Number of blocks in the WaveNet
    dropout = 0.1  # Dropout rate
    locations_kernel_size = 3  # Kernel size for the spatial dimension
    max_locations = 10  # Number of max_locations (for embedding)

    # Create a sample input tensor
    batch_size = 2  # Example batch size
    d_vars = 6  # Number of input features
    input_shape = (batch_size, seq_len, max_locations, d_vars)
    x = tf.random.normal(input_shape)

    # Location indices for embedding
    loc = tf.random.uniform((batch_size, max_locations), minval=0, maxval=max_locations, dtype=tf.int32)

    # Instantiate the SpatialEncoder
    encoder = SpatialEncoder(
        d_model=d_model,
        seq_len=seq_len,
        d_loc=max_locations,
        n_blocks=n_blocks,
        dropout=dropout,
        locations_kernel_size=locations_kernel_size
    )

    # Perform a forward pass
    output = encoder(x, loc, training=False)

    # Print the input and output shapes
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)