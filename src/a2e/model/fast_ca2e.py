import tensorflow as tf
from a2e.model.fast_a2e import FastA2E

@tf.keras.utils.register_keras_serializable(package="A2E")
class FastCA2E(FastA2E):
    def __init__(
            self,
            seq_len: int,
            lookback: int,
            d_model: int,
            max_locations: int,
            n_blocks: int,
            similarity_metric: str,
            k: int = False,
            key_stride: int = 1,
            dropout: float = 0,
            name="FastCA2E",
            *args,
            **kwargs
    ):
        """
        CA2E trained with "encode once, query many" (see FastA2E): the series of several sites are encoded once per
        step with their location embedding. With key_stride=1 the output is identical to CA2E for the same samples.
        """
        self.max_locations = max_locations  # Read by FastA2E.__init__ to build the conditional encoder
        super().__init__(
            seq_len=seq_len,
            lookback=lookback,
            d_model=d_model,
            n_blocks=n_blocks,
            similarity_metric=similarity_metric,
            k=k,
            key_stride=key_stride,
            dropout=dropout,
            name=name,
            *args,
            **kwargs
        )

    def _max_locations(self):
        """Include location embedding"""
        return self.max_locations

    def call(self, inputs) -> tf.Tensor:
        """
        :param inputs:
            - [0] x_s: Tensor with forecast series of all sites in this step (sites, time, d_vars)
            - [1] y_s: Tensor with observation series of all sites in this step (sites, time, 1)
            - [2] loc: Vector with location IDs of the sites (sites)
            - [3] idx: Tensor with indices into the encoded series (queries, 3):
                [:, 0] site within this step, [:, 1] encoded step of the query, [:, 2] encoded step of the first key
        :return: Tensor for aligned-output (see FastA2E)
        """
        x_s, y_s, loc, idx = inputs

        # Encode every site once with location conditioning
        z_s = self.encoder(x_s, loc=loc)            # shape (sites, time-seq_len+1, d_model)
        return self._retrieve(z_s, y_s, idx)

    def get_config(self):
        """Extract FastCA2E model configuration"""
        config = super().get_config()
        config.update({
            'max_locations': self.max_locations,
        })
        return config

    def build(self, input_shape):
        _, _, d_vars = input_shape[0]

        self.call([
            tf.zeros((1, self.lookback, d_vars)),
            tf.zeros((1, self.lookback, 1)),
            tf.zeros((1,), dtype=tf.int32),
            tf.zeros((1, 3), dtype=tf.int32),
        ])
        tf.keras.Model.build(self, input_shape)
