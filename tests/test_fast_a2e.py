import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tempfile
import tensorflow as tf
import a2e

# Small A2E setup on random data of a single location (time, variable)
tf.random.set_seed(0)
n_time = 600
forecasts = tf.random.normal((n_time, 2))
observations = tf.random.normal((n_time, 1))

def make_config(model_type, **kwargs):
    return a2e.io.ModelConfig(
        model_type=model_type,
        similarity_metric="cosine_similarity",
        d_model=8,
        n_blocks=2,
        seq_len=16,
        lookback=160,
        foresight=1,
        time_to_target=2,
        d_vars=2,
        dropout=0.3,
        **kwargs
    )

def build(config):
    factory = a2e.factory.factory.Factory(config)
    model = factory.build_model(optimizer=None, loss=None, metrics=None)
    pipeline = factory.build_pipeline(forecasts, observations, test_size=0.3, batch_size=8)
    return model, pipeline

a2e_model, a2e_pipeline = build(make_config("A2E"))
fast, fast_pipeline = build(make_config("FastA2E"))
fast.set_weights(a2e_model.get_weights())

# Normalized series as used inside the FastA2E pipeline
f_series, o_series = fast_pipeline._to_series(
    fast_pipeline.f_normalizer(forecasts),
    fast_pipeline.o_normalizer(observations)
)

########################################################################################################################
# 1) FastA2E reproduces A2E for the same samples (A2EPipeline orders samples by i)
########################################################################################################################
i = tf.constant([[0, 57, 211, 300, 391, 437]])
inputs, y_t = fast_pipeline._create_step(f_series, o_series, tf.constant([0]), i)
fast_output = fast(inputs, training=False)

samples = list(a2e_pipeline.full_dataset.as_numpy_iterator())
for q, sample_i in enumerate(i.numpy()[0]):
    (x_c, x_h, y_h), target = samples[sample_i]
    a2e_output = a2e_model((x_c[None], x_h[None], y_h[None]), training=False)
    tf.debugging.assert_near(fast_output[q], a2e_output[0], atol=1e-5)
    tf.debugging.assert_near(y_t[q], target)
print("OK: FastA2E output equals A2E output (values, weights and targets)")

########################################################################################################################
# 2) Training steps only contain the training span, test steps cover every test sample once
########################################################################################################################
(x_train, _, idx_train), _ = next(iter(fast_pipeline.get_train_data()))
assert x_train.shape[1] == fast_pipeline.train_span < n_time, x_train.shape
assert int(tf.reduce_max(idx_train[:, 1])) <= fast_pipeline.train_span - 16
test_i = [int(idx[0, 2]) for (_, _, idx), _ in fast_pipeline.get_test_data().take(fast_pipeline.get_test_steps())]
assert test_i[0] == fast_pipeline.num_train_per_location
assert sum(len(range(s, min(s + 8, fast_pipeline.num_samples_per_location))) for s in test_i) == fast_pipeline.test_size
print(f"OK: train steps use {fast_pipeline.train_span} of {n_time} time steps, test covers {fast_pipeline.test_size} samples")

########################################################################################################################
# 3) Key stride: every 3rd key, anchored at the most recent one
########################################################################################################################
strided = a2e.model.FastA2E(**{**fast.get_config(), "key_stride": 3})
strided.build(a2e.factory.factory.INPUT_SHAPE_CONFIGS["FastA2E"](make_config("FastA2E")))
strided.set_weights(fast.get_weights())
strided_output = strided(inputs, training=False)
n_keys = 160 - 16 + 1
assert strided_output.shape[1] == len(range(0, n_keys, 3)), strided_output.shape
tf.debugging.assert_near(strided_output[:, :, 0], tf.gather(fast_output[:, :, 0], strided.key_offsets, axis=1))
tf.debugging.assert_near(tf.reduce_sum(strided_output[:, :, 1], axis=1), tf.ones(i.shape[1]), atol=1e-5)
assert int(strided.key_offsets[-1]) == n_keys - 1
print(f"OK: key_stride=3 uses {strided_output.shape[1]} of {n_keys} keys")

########################################################################################################################
# 4) Training through the Api, saving, loading and embedding
########################################################################################################################
with tempfile.TemporaryDirectory() as directory:
    save_path = f"{directory}/fast_a2e.keras"
    api = a2e.io.Api(config=make_config("FastA2E", key_stride=2))
    model, history = api.train(
        forecasts=forecasts,
        observations=observations,
        save_path=save_path,
        epochs=2,
        batch_size=8,
        test_size=0.3,
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss=a2e.loss.SCRPS(),
        metrics=[a2e.metrics.keras.CRPSMetric()],
    )
    assert len(history.history["loss"]) == 2

    loaded = tf.keras.models.load_model(save_path, compile=False)
    tf.debugging.assert_near(loaded(inputs, training=False), model(inputs, training=False))
    assert api.embed(forecasts, verbose=False).shape == (n_time - 16 + 1, 8)
print("OK: Api.train, save/load and Api.embed work for FastA2E")
