import tensorflow as tf
from A2E.metrics.computation import (compute_cosine_similarity, compute_pearson_correlation,
                                     compute_scaled_dot_product, compute_euclidean_distance)

SIMILARITY_METRICS = {
    'cosine_similarity': compute_cosine_similarity,
    'pearson_correlation': compute_pearson_correlation,
    'scaled_dot_product': compute_scaled_dot_product,
    'euclidean_distance': compute_euclidean_distance,
}

@tf.keras.utils.register_keras_serializable(package="A2E")
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, similarity_metric: str, g_zero_init: int= None, name: str="cross_attention", *args, **kwargs):
        """
        Computes similarity between queries and keys, on a specified metric.
        Then applies a softmax to compute attention weights either on the full length of keys or just on top-k.
        """
        super().__init__(name=name, *args, **kwargs)
        self.similarity_metric = similarity_metric
        self.g_zero_init = tf.cast(g_zero_init, tf.float32)
        self.g_zero = None

        assert similarity_metric in list(SIMILARITY_METRICS.keys()),\
            ValueError(f"Unknown similarity metric: {similarity_metric}\n"
                      +f"Supported metrics: [{list(SIMILARITY_METRICS.values())}]")

        # Fetch similarity function based on given metric
        self.similarity_func = SIMILARITY_METRICS[similarity_metric]

        # Initialize g_zero
        if self.similarity_metric in ["cosine_similarity", "pearson_correlation"]:
            self.g_zero = float(
                tf.math.log(
                    tf.subtract(tf.square(self.g_zero_init), self.g_zero_init)
                ) / tf.math.log(tf.cast(2, tf.float32)))

    def build(self, input_shape):
        super().build(input_shape)
        # Henry et al. 2020: "Query-Key Normalization for Transformers"
        if self.similarity_metric in ["cosine_similarity", "pearson_correlation"]:
            self.g = self.add_weight(
                name='g',
                shape=(),
                initializer=tf.constant_initializer(self.g_zero),
                trainable=True,
                dtype=tf.float32
            )

    def call(self, query, keys, values, k=False):
        """
        Args:
            query: Query Tensor of shape (batch_size, 1, d_model)
            keys: Key Tensor of shape (batch_size, lookback, d_model)
            values: Value Tensor of shape (batch_size, lookback, 1)

        Returns:
            attention_scores: The computed attention-scores of shape (batch_size, lookback) or (batch_size, k)
        """
        similarity = self.similarity_func(query, keys)              # shape (batch, lookback)

        if self.similarity_metric in ["cosine_similarity", "pearson_correlation"]:
            similarity = tf.multiply(similarity, self.g) # Scales the similarity

        if k:
            similarity, indices = tf.math.top_k(similarity, k=k)    # shape (batch, k)
            values = tf.gather(values, indices, batch_dims=1)       # shape (batch, k)

        weights = tf.nn.softmax(similarity)                         # shape (batch, k)

        return values, weights
