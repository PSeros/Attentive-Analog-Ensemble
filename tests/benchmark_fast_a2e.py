import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import time
import tensorflow as tf
import a2e

# Same configuration as experiments/Trained_Models/A2E/A2E_Train.py, random data of one location
n_time, d_vars = 29925, 2
seq_len, lookback, time_to_target = 5 * (2 ** 4), 8760 * 2, 2
n_train = (n_time - lookback - time_to_target) - int(tf.math.ceil((n_time - lookback - time_to_target) * 0.3))
train_span = n_train + lookback + time_to_target  # time steps in a FastA2E training step
repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3

def make_model(model_type, **kwargs):
    config = a2e.io.ModelConfig(
        model_type=model_type,
        similarity_metric="cosine_similarity",
        d_model=32,
        n_blocks=5,
        seq_len=seq_len,
        lookback=lookback,
        foresight=1,
        time_to_target=time_to_target,
        d_vars=d_vars,
        dropout=0.3,
        **kwargs
    )
    return a2e.factory.factory.Factory(config).build_model(optimizer=None, loss=None, metrics=None)

def benchmark(model, inputs, target):
    loss = a2e.loss.SCRPS()

    @tf.function
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            value = loss(target, model(inputs, training=True))
        return tape.gradient(value, model.trainable_variables)

    train_step(inputs, target)  # trace
    start = time.perf_counter()
    for _ in range(repeats):
        gradients = train_step(inputs, target)
        _ = gradients[0].numpy()
    return (time.perf_counter() - start) / repeats

# A2E: every sample carries and encodes its own lookback
batch = 64
a2e_inputs = (
    tf.random.normal((batch, seq_len, d_vars)),
    tf.random.normal((batch, lookback, d_vars)),
    tf.random.normal((batch, lookback, 1)),
)
a2e_time = benchmark(make_model("A2E"), a2e_inputs, tf.random.normal((batch,)))
print(f"A2E      batch={batch:4d}            : {a2e_time:7.3f} s/step  {a2e_time / batch * 1e3:8.2f} ms/query")

# FastA2E: the training span of the series once, queries index into the shared embeddings
for queries, key_stride in [(64, 1), (256, 1), (64, 3), (256, 3)]:
    i = tf.random.uniform((queries,), maxval=n_train, dtype=tf.int32)
    fast_inputs = (
        tf.random.normal((1, train_span, d_vars)),
        tf.random.normal((1, train_span, 1)),
        tf.stack([tf.zeros_like(i), i + lookback + time_to_target - seq_len + 1, i], axis=-1),
    )
    fast_time = benchmark(make_model("FastA2E", key_stride=key_stride), fast_inputs, tf.random.normal((queries,)))
    print(f"FastA2E  queries={queries:4d} stride={key_stride} : "
          f"{fast_time:7.3f} s/step  {fast_time / queries * 1e3:8.2f} ms/query  "
          f"({(a2e_time / batch) / (fast_time / queries):5.0f}x faster per query)")
