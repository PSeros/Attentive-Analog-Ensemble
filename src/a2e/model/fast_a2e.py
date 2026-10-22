import tensorflow as tf
from a2e.layer.encoder import Encoder
from a2e.layer.cross_attention import CrossAttention

@tf.keras.utils.register_keras_serializable(package="A2E")
class FastA2E(tf.keras.Model):
    def __init__(
            self,
            seq_len: int,
            lookback: int,
            d_model: int,
            n_blocks: int,
            similarity_metric: str,
            k: int = False,
            key_stride: int = 1,
            dropout: float = 0,
            name="FastA2E",
            *args,
            **kwargs
    ):
        """
        A2E trained with "encode once, query many":
        Instead of encoding the lookback of every sample separately, the full series is encoded once per step and
        keys are rows of these shared embeddings, gathered by index. Queries are encoded from their own windows, so
        that each query gets its own dropout mask, independent of its keys, as in A2E.
        Uses the same layers as A2E, with key_stride=1 the output is identical to A2E for the same samples.
        """
        super().__init__(name=name, *args, **kwargs)
        self.seq_len = seq_len
        self.lookback = lookback
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.similarity_metric = similarity_metric
        self.k = k
        self.key_stride = key_stride
        self.dropout = dropout

        # Key offsets relative to the first key of a lookback, anchored at the most recent key
        n_keys = lookback - seq_len + 1
        self.key_offsets = tf.range(n_keys - 1, -1, -key_stride)[::-1]  # shape (n_strided_keys,)

        # Encoder
        self.encoder = Encoder(
            d_model=d_model,
            seq_len=seq_len,
            n_blocks=n_blocks,
            dropout=dropout,
            max_locations=self._max_locations(),
            name="encoder"
        )

        # Cross-Attention (g_zero_init is only applied for ["cosine_similarity", "pearson_correlation"])
        self.cross_attention = CrossAttention(similarity_metric, g_zero_init=len(self.key_offsets), name="cross_attention")

    def _max_locations(self):
        """Number of location embeddings of the encoder. None -> no conditional encoding for A2E"""
        return None

    def call(self, inputs, training=None) -> tf.Tensor:
        """
        :param inputs:
            - [0] x_s: Tensor with forecast series (series, time, d_vars)
            - [1] y_s: Tensor with observation series (series, time, 1)
            - [2] idx: Tensor with indices into the encoded series (queries, 3):
                [:, 0] series, [:, 1] encoded step of the query, [:, 2] encoded step of the first key
        :param training: Enables dropout. Keras 3 fit() passes training=True only if call() accepts it.
        :return: Tensor for aligned-output:
            - [:, :, 0] y_h: Tensor with historical observations (queries, n_strided_keys) or (queries, k)
            - [:, :, 1] w_h: Tensor with historical weights (queries, n_strided_keys) or (queries, k)
        """
        x_s, y_s, idx = inputs
        idx = tf.cast(idx, tf.int32)

        # Encode every series once -> encoded step j holds the window ending at time step j + seq_len - 1
        z_s = self.encoder(x_s, training=training)  # shape (series, time-seq_len+1, d_model)

        # Encode query windows separately -> own dropout mask per query
        z_c = self.encoder(self._query_windows(x_s, idx), training=training)  # shape (queries, 1, d_model)
        return self._retrieve(z_c, z_s, y_s, idx)

    def _query_windows(self, x_s, idx) -> tf.Tensor:
        """Cuts the window of every query out of the series -> shape (queries, seq_len, d_vars)"""
        steps = idx[:, 1:2] + tf.range(self.seq_len)
        series = tf.broadcast_to(idx[:, :1], tf.shape(steps))
        return tf.gather_nd(x_s, tf.stack([series, steps], axis=-1))

    def _retrieve(self, z_c, z_s, y_s, idx) -> tf.Tensor:
        """Gathers keys and values from the encoded series and applies the cross-attention to the queries."""
        # Align y_s with z_s (remove padded region)
        y_s = y_s[:, self.seq_len - 1:, 0]          # shape (series, time-seq_len+1)

        # Gather keys and values of each query's lookback
        key_steps = idx[:, 2:3] + self.key_offsets  # shape (queries, n_strided_keys)
        series = tf.broadcast_to(idx[:, :1], tf.shape(key_steps))
        key_idx = tf.stack([series, key_steps], axis=-1)
        z_h = tf.gather_nd(z_s, key_idx)            # shape (queries, n_strided_keys, d_model)
        y_h = tf.gather_nd(y_s, key_idx)            # shape (queries, n_strided_keys)

        # Cross-Attention -> Attention Weights, Values | both with shape (queries, n_strided_keys) or (queries, k)
        y_h, w_h = self.cross_attention(query=z_c, keys=z_h, values=y_h, k=self.k)

        # Merge Outputs -> shape (queries, n_strided_keys, 2) or (queries, k, 2)
        y_h = tf.expand_dims(y_h, axis=-1)          # shape (queries, None, 1)
        w_h = tf.expand_dims(w_h, axis=-1)          # shape (queries, None, 1)
        merged = tf.concat([y_h, w_h], axis=-1)
        return merged

    def get_config(self):
        """Extract FastA2E model configuration"""
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'lookback': self.lookback,
            'd_model': self.d_model,
            'n_blocks': self.n_blocks,
            'similarity_metric': self.similarity_metric,
            'k': self.k,
            'key_stride': self.key_stride,
            'dropout': self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create FastA2E instance from config"""
        return cls(**config)

    def compute_output_shape(self, input_shape):
        batch = input_shape[-1][0]
        output_steps = self.k if self.k else len(self.key_offsets)
        return (batch, output_steps, 2)

    def build(self, input_shape):
        _, _, d_vars = input_shape[0]

        self.call([
            tf.zeros((1, self.lookback, d_vars)),
            tf.zeros((1, self.lookback, 1)),
            tf.zeros((1, 3), dtype=tf.int32),
        ])
        super().build(input_shape)
