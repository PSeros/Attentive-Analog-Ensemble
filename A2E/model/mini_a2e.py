import tensorflow as tf
from A2E.model.base import BaseModel
from A2E.layer.cross_attention import CrossAttention

@tf.keras.utils.register_keras_serializable(package="A2E")
class MiniA2E(BaseModel):
    def __init__(
            self,
            seq_len: int,
            lookback: int,
            d_model: int,
            n_blocks: int,
            similarity_metric: str,
            k: int = False,
            dropout: float = 0,
            name="MiniA2E",
            *args,
            **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.seq_len = seq_len
        self.lookback = lookback
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.similarity_metric = similarity_metric
        self.k = k
        self.dropout = dropout

        # Encoder
        self.encoder = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=seq_len,
            name="encoder"
        )

        # Cross-Attention (g_zero_init is only applied for ["cosine_similarity", "pearson_correlation"])
        self.cross_attention = CrossAttention(similarity_metric, g_zero_init=lookback-seq_len+1, name="cross_attention")

    def call(self, inputs) -> tf.Tensor:
        """
        :param inputs:
            - [0] x_c: Tensor with current forecasts  (batch, seq_len, d_vars)
            - [1] x_h: Tensor with historical forecasts (batch, lookback, d_vars)
            - [2] y_h: Tensor with historical observations (batch, lookback, 1)
        :return: Tensor for aligned-output:
            - [:, :, 0] Y_h: Tensor with historical observations (batch, lookback)
            - [:, :, 1] W_h: Tensor with historical weights (batch, lookback)
        """
        x_c, x_h, y_h = inputs

        # Encode both current and historical data with siamese encoder
        z_c = self.encoder(x_c)  # shape (batch, seq_len, d_model)
        z_h = self.encoder(x_h)  # shape (batch, lookback, d_model)

        # Align y_h (remove padded region)
        y_h = y_h[:, self.seq_len - 1:, 0]          # shape (batch, new_lookback)

        # Cross-Attention -> Attention Weights, Values | both with shape (batch, new_lookback) or (batch, k)
        y_h, w_h = self.cross_attention(query=z_c, keys=z_h, values=y_h, k=self.k)

        # Merge Outputs -> shape (batch, lookback, 2) or (batch, k, 2)
        y_h = tf.expand_dims(y_h, axis=-1)          # shape (batch, None, 1)
        w_h = tf.expand_dims(w_h, axis=-1)          # shape (batch, None, 1)
        merged = tf.concat([y_h, w_h], axis=-1)
        return merged

    def get_config(self):
        """Extract A2E model configuration"""
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'lookback': self.lookback,
            'd_model': self.d_model,
            'n_blocks': self.n_blocks,
            'similarity_metric': self.similarity_metric,
            'dropout': self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create A2E instance from config"""
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)
        _, self.lookback, d_vars = input_shape[1]

        # Build encoder for both current and historical data -> TimeDim = None
        self.encoder.build((None, None, d_vars))

        # Build cross-attention
        self.cross_attention.build((
            (None, 1, self.d_model),
            (None, None, self.d_model),
            (None, None, 1),
        ))