import tensorflow as tf
from a2e.metrics.computation import (compute_cosine_similarity, compute_pearson_correlation,
                                     compute_scaled_dot_product, compute_euclidean_distance)

SIMILARITY_METRICS = {
    'cosine_similarity': compute_cosine_similarity,
    'pearson_correlation': compute_pearson_correlation,
    'scaled_dot_product': compute_scaled_dot_product,
    'euclidean_distance': compute_euclidean_distance,
}

# Bounded similarities, scaled by the learned temperature g
TEMPERATURE_METRICS = ["cosine_similarity", "pearson_correlation"]

# Parametrizations of g, one trainable weight each (ModelConfig.g_param):
#   "g":   scores = g · s, g starts at g_0 (submitted models)
#   "tau": scores = s / τ, τ = 1/g starts at 1/g_0
G_PARAMS = ("g", "tau")

@tf.keras.utils.register_keras_serializable(package="A2E")
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, similarity_metric: str, g_zero_init: int= None, g_param: str = "g",
                 name: str="cross_attention", *args, **kwargs):
        """
        Computes similarity between queries and keys, on a specified metric.
        Then applies a softmax to compute attention weights either on the full length of keys or just on top-k.

        Args:
            similarity_metric: one of SIMILARITY_METRICS
            g_zero_init: archive length N; the temperature g starts at g_0 = log2(N² − N)
                (only for TEMPERATURE_METRICS)
            g_param: trainable weight behind g (G_PARAMS)
                "g": g itself (submitted models). An Adam step changes g by at most about the learning rate, which
                    is very little relative to g_0 (≈ 28 for a two-year archive).
                "tau": the similarity is divided by τ = 1/g, τ starts at 1/g_0. The same step on τ changes g by
                    about lr · g relatively, so g can adapt during training.
        """
        super().__init__(name=name, *args, **kwargs)
        self.similarity_metric = similarity_metric
        self.g_zero_init = tf.cast(g_zero_init, tf.float32)
        self.g_zero = None
        if g_param not in G_PARAMS:
            raise ValueError(f"Unknown g_param: {g_param}. Supported: {G_PARAMS}")
        self.g_param = g_param
        self.g_weight = None    # trainable weight behind g: g or τ = 1/g (g_param)

        assert similarity_metric in list(SIMILARITY_METRICS.keys()),\
            ValueError(f"Unknown similarity metric: {similarity_metric}\n"
                      +f"Supported metrics: [{list(SIMILARITY_METRICS.values())}]")

        # Fetch similarity function based on given metric
        self.similarity_func = SIMILARITY_METRICS[similarity_metric]

        # Initialize g_zero
        if self.similarity_metric in TEMPERATURE_METRICS:
            self.g_zero = float(
                tf.math.log(
                    tf.subtract(tf.square(self.g_zero_init), self.g_zero_init)
                ) / tf.math.log(tf.cast(2, tf.float32)))

    def build(self, input_shape):
        super().build(input_shape)
        # Henry et al. 2020: "Query-Key Normalization for Transformers"
        # One weight in both variants: the saved weights of the layer are matched by position, not by name.
        if self.similarity_metric in TEMPERATURE_METRICS:
            initial = 1.0 / self.g_zero if self.g_param == "tau" else self.g_zero
            self.g_weight = self.add_weight(
                name=self.g_param,
                shape=(),
                initializer=tf.constant_initializer(initial),
                trainable=True,
                dtype=tf.float32
            )

    @property
    def g(self):
        """Temperature g: the weight itself, or 1/τ (g_param). AttributeError for metrics without g."""
        if self.g_weight is None:
            raise AttributeError(f"No temperature g for {self.similarity_metric} (or layer not built).")
        if self.g_param == "tau":
            return tf.math.reciprocal(self.g_weight)
        return self.g_weight

    def scores(self, query, keys):
        """
        Attention logits: similarity between query and keys, scaled by the learned temperature g
        (cosine similarity, pearson correlation).

        Args:
            query: Query Tensor of shape (batch_size, 1, d_model)
            keys: Key Tensor of shape (batch_size, lookback, d_model)

        Returns:
            similarity of shape (batch_size, lookback)
        """
        similarity = self.similarity_func(query, keys)              # shape (batch, lookback)
        if self.similarity_metric in TEMPERATURE_METRICS:
            if self.g_param == "tau":
                similarity = tf.divide(similarity, self.g_weight)
            else:
                similarity = tf.multiply(similarity, self.g) # Scales the similarity
        return similarity

    def call(self, query, keys, values, k=False):
        """
        Args:
            query: Query Tensor of shape (batch_size, 1, d_model)
            keys: Key Tensor of shape (batch_size, lookback, d_model)
            values: Value Tensor of shape (batch_size, lookback, 1)

        Returns:
            attention_scores: The computed attention-scores of shape (batch_size, lookback) or (batch_size, k)
        """
        similarity = self.scores(query, keys)                       # shape (batch, lookback)

        if k:
            similarity, indices = tf.math.top_k(similarity, k=k)    # shape (batch, k)
            values = tf.gather(values, indices, batch_dims=1)       # shape (batch, k)

        weights = tf.nn.softmax(similarity)                         # shape (batch, k)

        return values, weights
