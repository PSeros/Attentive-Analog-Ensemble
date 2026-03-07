import tensorflow as tf

########################################################################################################################
# Probabilistic Forcast Metrics
########################################################################################################################
def _term1(x: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
    "E|X - X'|"
    x_sorted = tf.sort(x, axis=1)
    sort_indices = tf.argsort(x, axis=1)
    w_sorted = tf.gather(w, sort_indices, batch_dims=1)
    wx_sorted = tf.multiply(w_sorted, x_sorted)

    w_cum = tf.cumsum(w_sorted, axis=1, exclusive=True)
    wx_cum = tf.cumsum(wx_sorted, axis=1, exclusive=True)

    x_w_cum = tf.multiply(x_sorted, w_cum)
    x_w_dif = tf.subtract(x_w_cum, wx_cum)
    dif_contrib = tf.multiply(w_sorted, x_w_dif)
    return tf.multiply(tf.cast(2, tf.float32), tf.reduce_sum(dif_contrib, axis=1))

def _term2(x: tf.Tensor, w: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    "E|X - y|"
    abs_diff = tf.abs(tf.subtract(x,y))
    return tf.reduce_sum(tf.multiply(abs_diff, w), axis=1)

def compute_crps(x: tf.Tensor, w: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Computes CRPS with non-uniform but normalized weights. In O(n log n)"""
    if tf.rank(y) ==1:
        y = tf.expand_dims(y, -1)
    t1 = _term1(x, w)
    t2 = _term2(x, w, y)
    return  tf.subtract(tf.multiply(tf.cast(0.5, tf.float32),t1) ,t2)

def compute_scrps(x: tf.Tensor, w: tf.Tensor, y: tf.Tensor, gamma: float=1e-8) -> tf.Tensor:
    """
    Computes scaled CRPS -> SCRPS (Bolin; Wallin (2022) "LOCAL SCALE INVARIANCE AND ROBUSTNESS OF PROPER SCORING RULES")
    with non-uniform but normalized weights. In O(n log n)
    """
    if tf.rank(y) ==1:
        y = tf.expand_dims(y, -1)
    gamma = tf.cast(gamma, tf.float32)
    t1 = tf.add(_term1(x, w), gamma) # gamma -> prevent from division/log of 0
    t2 = _term2(x, w, y)
    return tf.subtract(-tf.divide(t2,t1), tf.multiply(tf.cast(0.5, tf.float32),tf.math.log(t1)))

########################################################################################################################
# Deterministic Forcast Metrics
########################################################################################################################
def compute_bias(x: tf.Tensor, w: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Compute the bias (absolute error) between the ensemble mean and true values.

    Parameters
    ----------
    x : tf.Tensor
        Ensemble observations, shape (n_samples, n_members)
    w : tf.Tensor
        Weights for ensemble members, shape (n_samples, n_members). Must be normalized so that sum(W, axis=1) == 1.
    y : tf.Tensor
        True values, shape (n_samples,)

    Returns
    -------
    tf.Tensor
        Array of bias values, shape (n_samples,)
    """
    # Ensure broadcastable shape shape (n_samples, 1)
    if tf.rank(y) == 1:
        y = tf.expand_dims(y, axis=1)
    # Calculate ensemble mean
    expected_values = tf.reduce_sum(tf.multiply(x, w), axis=1, keepdims=True)
    # Calculate Bias
    bias = tf.subtract(expected_values, y)
    return bias

def compute_rmse(x: tf.Tensor, w: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Compute the RMSE between ensemble members and true values for each time step.

    Parameters
    ----------
    X : tf.Tensor
        Ensemble observations, shape (n_samples, n_members)
    W : tf.Tensor
        Weights for ensemble members, shape (n_samples, n_members). Must be normalized so that sum(W, axis=1) == 1.
    y : tf.Tensor
        True values, shape (n_samples,)

    Returns
    -------
    tf.Tensor
        Array of RMSE values, shape (n_samples,)
    """
    # Ensure broadcastable shape shape (n_samples, 1)
    if tf.rank(y) == 1:
        y = tf.expand_dims(y, axis=1)
    # Calculate squared errors
    squared_errors = tf.square(tf.subtract(x, y))
    # Calculate weighted MSE
    weighted_mse = tf.reduce_sum(tf.multiply(w, squared_errors), axis=1)
    # Calculate RMSE
    rmse_values = tf.sqrt(weighted_mse)
    return rmse_values

########################################################################################################################
# Similarity metrics
########################################################################################################################
def compute_cosine_similarity(current: tf.Tensor, historical: tf.Tensor) -> tf.Tensor:
    """Computes the cosine similarity between current and historical embeddings."""
    current = tf.nn.l2_normalize(current, axis=-1)
    historical = tf.nn.l2_normalize(historical, axis=-1)
    similarity = tf.matmul(current, historical, transpose_b=True)
    similarity = tf.squeeze(similarity, 1)
    return similarity

def compute_pearson_correlation(current: tf.Tensor, historical: tf.Tensor) -> tf.Tensor:
    """Computes the pearson correlation between current and historical"""
    current = tf.subtract(current, tf.reduce_mean(current, axis=-1, keepdims=True))
    historical = tf.subtract(historical, tf.reduce_mean(historical, axis=-1, keepdims=True))
    current = tf.nn.l2_normalize(current, axis=-1)
    historical = tf.nn.l2_normalize(historical, axis=-1)
    similarity = tf.matmul(current, historical, transpose_b=True)
    similarity = tf.squeeze(similarity, 1)
    return similarity

def compute_scaled_dot_product(current: tf.Tensor, historical: tf.Tensor) -> tf.Tensor:
    """Computes the scaled dot product between current and historical"""
    similarity = tf.matmul(current, historical, transpose_b=True)
    similarity = tf.divide(similarity, tf.sqrt(tf.cast(current.get_shape()[-1], tf.float32)))
    similarity = tf.squeeze(similarity, 1)
    return similarity

def compute_euclidean_distance(current: tf.Tensor, historical: tf.Tensor, eps: float=1e-8) -> tf.Tensor:
    """Computes the negative Euclidean distance between current and historical"""
    dif = tf.subtract(current, historical)
    squared_dif = tf.square(dif)
    summed_squared_dif = tf.reduce_sum(squared_dif, axis=-1)
    return -tf.sqrt(summed_squared_dif + eps)

########################################################################################################################
# others
########################################################################################################################
def compute_entropy(w: tf.Tensor, eps: float=1e-8) -> tf.Tensor:
    eps = tf.cast(eps, tf.float32)
    log_w = tf.math.log(tf.add(w, eps))
    wlog_w = tf.multiply(w, log_w)
    return -tf.reduce_sum(wlog_w, axis=-1)

def compute_cross_entropy(w: tf.Tensor, eps: float=1e-8) -> tf.Tensor:
    eps = tf.cast(eps, tf.float32)
    log_w = tf.math.log(tf.add(w, eps))
    return -tf.reduce_mean(log_w, axis=-1)

def compute_quantiles(values: tf.Tensor, weights: tf.Tensor, quantiles: tf.Tensor) -> tf.Tensor:
    """Computes weighted Quantiles."""
    indices  = tf.argsort(values, axis=1)
    v_sorted = tf.gather(values, indices, axis=1, batch_dims=1)
    w_sorted = tf.gather(weights, indices, axis=1, batch_dims=1)
    cw = tf.cumsum(w_sorted, axis=1)
    # Identify the idx where cumulative w is >= specified quantiles
    idx = tf.argmax(tf.cast(cw[:, :, None] >= quantiles[None, None, :], tf.int32), axis=1)
    return tf.gather(v_sorted, idx, batch_dims=1)