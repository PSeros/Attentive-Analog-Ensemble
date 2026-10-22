import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import time
import tensorflow as tf
import a2e

# Same configuration as experiments/Trained_Models/CA2E/CA2E_Train.py, random data of the same size
n_time, n_locations, d_vars = 29925, 82, 2
seq_len, lookback, time_to_target = 5 * (2 ** 4), 8760 * 2, 2
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
        max_locations=n_locations,
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

# CA2E: every sample carries and encodes its own lookback
batch = 64
ca2e_inputs = (
    tf.random.normal((batch, seq_len, d_vars)),
    tf.random.normal((batch, lookback, d_vars)),
    tf.random.normal((batch, lookback, 1)),
    tf.random.uniform((batch,), maxval=n_locations, dtype=tf.int32),
)
ca2e_time = benchmark(make_model("CA2E"), ca2e_inputs, tf.random.normal((batch,)))
print(f"CA2E      batch={batch:4d}                    : {ca2e_time:7.3f} s/step  {ca2e_time / batch * 1e3:8.2f} ms/query")

# FastCA2E: full series of a few sites, queries index into the shared embeddings
for sites_per_step, queries, key_stride in [(1, 64, 1), (4, 256, 1), (4, 256, 3), (8, 1024, 3)]:
    i = tf.random.uniform((queries,), maxval=n_time - lookback - time_to_target, dtype=tf.int32)
    site = tf.random.uniform((queries,), maxval=sites_per_step, dtype=tf.int32)
    fast_inputs = (
        tf.random.normal((sites_per_step, n_time, d_vars)),
        tf.random.normal((sites_per_step, n_time, 1)),
        tf.range(sites_per_step),
        tf.stack([site, i + lookback + time_to_target - seq_len + 1, i], axis=-1),
    )
    fast_time = benchmark(make_model("FastCA2E", key_stride=key_stride), fast_inputs, tf.random.normal((queries,)))
    print(f"FastCA2E  sites={sites_per_step} queries={queries:4d} stride={key_stride} : "
          f"{fast_time:7.3f} s/step  {fast_time / queries * 1e3:8.2f} ms/query  "
          f"({(ca2e_time / batch) / (fast_time / queries):5.0f}x faster per query)")
