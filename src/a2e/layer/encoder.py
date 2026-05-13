import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="A2E")
class DilatedConvLayer(tf.keras.layers.Layer):
    """
    Dilated Convolutional layer for 3D tensors (batch, time, d_model)
    A Causal temporal dilated convolution for temporal data.
    Supports conditional processing similar to the conditional WaveNet.
    """
    def __init__(self, filter, seq_len, dropout=.0, name=None, **kwargs):
        """
        Initializes a dilated convolutional layer for use in a WaveNet model. Using Gated Activation Units.

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
        kernel_size = 2
        _seq_len = tf.constant(self.seq_len, dtype=tf.float32)
        _kernel_size = tf.constant(kernel_size, dtype=tf.float32)
        _num_layers = tf.math.log(_seq_len) / tf.math.log(_kernel_size)
        num_layers = int(tf.math.ceil(_num_layers))

        # Dilated causal convolution W
        self.W_f = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                filters=self.filter,
                kernel_size=kernel_size,
                dilation_rate=kernel_size**i,
                padding='causal',
                use_bias=False
            ) for i in range(num_layers)]
        )
        self.W_g = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                filters=self.filter,
                kernel_size=kernel_size,
                dilation_rate=kernel_size**i,
                padding='causal',
                use_bias=False
            ) for i in range(num_layers)]
        )

        # Conditional Bias V|h (only used when h is provided)
        self.V_f = tf.keras.layers.Dense(self.filter, use_bias=False)
        self.V_g = tf.keras.layers.Dense(self.filter, use_bias=False)

        # Regularization
        self.dropout = tf.keras.layers.Dropout(
            rate=dropout,
            noise_shape=(None, 1, self.filter)
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
            x (tensor): Input tensor for the layer with shape(batch_size, time, d_model).
            h (tensor, optional): Conditional input tensor with shape(batch_size, d_model). Defaults to None.
            training (bool, optional): Whether the model is in training mode or not. Defaults to None.

        Returns:
            tensor: The output tensor after applying dilated convolutions, conditional biases,
                   gated activation, and dropout, with shape (batch_size, time, d_model).
        """
        Wx_f = self.W_f(x)
        Wx_g = self.W_g(x)

        if h is not None:
            # Conditional case: add conditioning signal
            Vh_f = self.V_f(h)
            Vh_g = self.V_g(h)
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


@tf.keras.utils.register_keras_serializable(package="A2E")
class WaveNetBlock(tf.keras.layers.Layer):
    """
    WaveNet Block for WaveNet.
    """
    def __init__(self, d_model, seq_len, dropout=.0, name=None, **kwargs):
        """
        Initializes a block used in the WaveNet architecture.
        Consisting of a Dilated Convolution and a linear projection.
        Including Residual and Scip connections.

        Args:
            d_model (int): Dimensionality of the model, corresponding to the number of filters.
            seq_len (int): Length of the input sequence.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.
            name (str, optional): Name of the layer. Defaults to None.
            **kwargs: Additional keyword arguments for parent class initialization.
        """
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        self.dilated_conv = DilatedConvLayer(
            filter=d_model,
            seq_len=seq_len,
            dropout=dropout,
        )
        self.dense = tf.keras.layers.Dense(d_model)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            x = input_shape[0]
        else:
            x = input_shape
        return (x, x)

    def call(self, residual, h=None, skip=0, training=None):
        """
        Applies the WaveNet block operations to the residual tensor and updates skip connections.

        Args:
            residual (tensor): The input residual tensor with shape(batch_size, time, d_model).
            h (tensor, optional): Conditional input tensor with shape(batch_size, d_model). Defaults to None.
            skip (tensor, optional): The skip connection tensor from previous blocks with shape(batch_size, seq_len, d_model). Defaults to 0.
            training (bool, optional): Whether the model is in training mode or not. Defaults to None.

        Returns:
            tuple: A tuple containing the updated residual tensor with shape(batch_size, time, d_model)
                  and the updated skip connection tensor with shape(batch_size, time, d_model).
        """
        dilated_conv_output = self.dilated_conv(residual, h, training=training)
        dense_output = self.dense(dilated_conv_output)

        skip = tf.add(skip, dense_output)
        residual = tf.add(residual, dense_output)

        return residual, skip

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "seq_len": self.seq_len,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="A2E")
class WaveNet(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, n_blocks, dropout=.0, name="WaveNet", **kwargs):
        """
        Initializes a WaveNet model with multiple WaveNet blocks,
        and two final Dense Layers (first with ReLu and second Linear) which output the final embedding z.

        Args:
            d_model (int): Dimensionality of the model.
            seq_len (int): Length of the input sequence.
            n_blocks (int): Number of blocks in the WaveNet.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.
            name (str, optional): Name of the layer. Defaults to "WaveNet".
            **kwargs: Additional keyword arguments for parent class initialization.
        """
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        self.dropout = dropout

        # Key Components
        self.wave_net_blocks = [WaveNetBlock(
            d_model=d_model,
            seq_len=seq_len//n_blocks,
            dropout=dropout,
        ) for _ in range(n_blocks)]
        self.dense1 = tf.keras.layers.Dense(d_model, activation="relu")
        self.dense2 = tf.keras.layers.Dense(d_model, activation=None)

        # Regularization
        self.dropout = tf.keras.layers.Dropout(
            rate=dropout,
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
        Applies multiple WaveNet blocks to the input tensor and processes the result.

        Args:
            x (tensor): Input tensor for the model with shape(batch_size, time, d_model).
            h (tensor, optional): Conditional input tensor with shape(batch_size, d_model). Defaults to None.
            training (bool, optional): Whether the model is in training mode or not. Defaults to None.

        Returns:
            tensor: The output tensor after processing through WaveNet blocks, dense layers, and dropout,
                   with shape(batch_size, time-seq_len+1, d_model).
        """
        residual = x
        skip = tf.zeros_like(x)

        for block in self.wave_net_blocks:
            residual, skip = block(residual, h, skip, training=training)

        # Cropping padded regions
        skip = skip[:, self.seq_len-1:, :]

        # Final transformations
        skip = tf.nn.relu(skip)
        skip = self.dense1(skip)
        skip = self.dropout(skip, training=training)
        z = self.dense2(skip)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "seq_len": self.seq_len,
            "n_blocks": self.n_blocks,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="A2E")
class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, n_blocks, dropout=.0, max_locations=None, name="Encoder", **kwargs):
        """
        Initializes an encoder layer that projects inputs, uses conditional location embeddings and encodes it with
        a WaveNet model.

        Args:
            d_model (int): Dimensionality of the model.
            seq_len (int): Length of the input sequence.
            n_blocks (int): Number of blocks in the WaveNet.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.
            max_locations (int, optional): Number of location embeddings. Defaults to None.
            name (str, optional): Name of the layer. Defaults to "Encoder".
            **kwargs: Additional keyword arguments for parent class initialization.
        """
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.max_locations = max_locations # Initialize as None by default

        # Projection
        self.projection = tf.keras.layers.Dense(d_model, name="projection")

        # Embedding (for Conditional Encoder)
        self.embedding = None
        if max_locations:
            self.embedding = tf.keras.layers.Embedding(
                input_dim=max_locations,
                output_dim=d_model,
                name="loc_embedding"
            )

        # WaveNet
        self.wave_net = WaveNet(
            d_model=d_model,
            seq_len=seq_len,
            n_blocks=n_blocks,
            dropout=dropout,
            name="wavenet"
        )

    def build(self, input_shape):
        super().build(input_shape)
        _, _, d_vars = input_shape

        # Build layers
        self.projection.build((None, None, d_vars))
        self.wave_net.build([(None, None, self.d_model), (None, self.d_model)])

        if self.embedding:
            self.embedding.build((None,))

    def call(self, x, loc=None, training=None):
        """
        Projects the input tensor and encodes it using location embeddings and a WaveNet model.

        Args:
            x (tensor): Input tensor for the encoder with shape(batch_size, time, d_vars).
            loc (tensor, optional): Location input tensor for conditional embedding with shape(batch_size). Defaults to None.
            training (bool, optional): Flag indicating whether the layer is in training mode.

        Returns:
            tensor: The encoded output tensor with shape(batch_size, time-seq_len+1, d_model).
        """
        # Project to d_model
        x = self.projection(x)

        # Get location embedding if used
        h = None
        if loc is not None and self.embedding is not None:
            h = self.embedding(loc)

        # Apply WaveNet encoding
        z = self.wave_net(x, h, training=training)
        return z

    def get_config(self):
        """Extract configuration"""
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'max_locations': self.max_locations,
            'n_blocks': self.n_blocks,
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
    d_model = 64  # Dimension of the model
    seq_len = 10  # Length of the input sequence
    n_blocks = 5  # Number of blocks in the WaveNet
    dropout = 0.1  # Dropout rate

    # Create a sample input tensor
    batch_size = 2  # Example batch size
    d_vars = 16  # Number of input features
    input_shape = (batch_size, seq_len, d_vars)
    x = tf.random.normal(input_shape)

    # Instantiate the SpatialEncoder
    encoder = Encoder(
        d_model=d_model,
        seq_len=seq_len,
        n_blocks=n_blocks,
        dropout=dropout,
    )

    # Perform a forward pass
    output = encoder(x, training=False)

    # Print the input and output shapes
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)