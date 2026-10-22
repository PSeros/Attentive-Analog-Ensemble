import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tempfile
import tensorflow as tf
import a2e

# Small CA2E setup on random data (time, location, variable)
tf.random.set_seed(0)
n_time, n_locations = 600, 5
forecasts = tf.random.normal((n_time, n_locations, 2))
observations = tf.random.normal((n_time, n_locations, 1))

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
        max_locations=n_locations,
        dropout=0.3,
        **kwargs
    )

def build(config):
    factory = a2e.factory.factory.Factory(config)
    model = factory.build_model(optimizer=None, loss=None, metrics=None)
    pipeline = factory.build_pipeline(forecasts, observations, test_size=0.3, batch_size=8)
    return model, pipeline

ca2e, ca2e_pipeline = build(make_config("CA2E"))
fast, fast_pipeline = build(make_config("FastCA2E", sites_per_step=2))
fast.set_weights(ca2e.get_weights())

# Normalized site series as used inside the FastCA2E pipeline
f_sites = tf.transpose(fast_pipeline.f_normalizer(forecasts), [1, 0, 2])
o_sites = tf.transpose(fast_pipeline.o_normalizer(observations), [1, 0, 2])

########################################################################################################################
# 1) FastCA2E reproduces CA2E for the same samples (CA2EPipeline orders samples as i * n_locations + loc)
########################################################################################################################
sites = tf.constant([3, 1])
i = tf.constant([[0, 57, 211, 430], [5, 99, 300, 437]])
(x_s, y_s, loc, idx), y_t = fast_pipeline._create_step(f_sites, o_sites, sites, i)
fast_output = fast((x_s, y_s, loc, idx), training=False)

samples = list(ca2e_pipeline.full_dataset.as_numpy_iterator())
for q, (s, sample_i) in enumerate([(s, int(j)) for s, row in zip(sites.numpy(), i.numpy()) for j in row]):
    (x_c, x_h, y_h, l), target = samples[sample_i * n_locations + s]
    ca2e_output = ca2e((x_c[None], x_h[None], y_h[None], l[None]), training=False)
    tf.debugging.assert_near(fast_output[q], ca2e_output[0], atol=1e-5)
    tf.debugging.assert_near(y_t[q], target)
print("OK: FastCA2E output equals CA2E output (values, weights and targets)")

########################################################################################################################
# 2) Key stride: every 3rd key, anchored at the most recent one
########################################################################################################################
strided = a2e.model.FastCA2E(**{**fast.get_config(), "key_stride": 3})
strided.build(a2e.factory.factory.INPUT_SHAPE_CONFIGS["FastCA2E"](make_config("FastCA2E")))
strided.set_weights(fast.get_weights())
strided_output = strided((x_s, y_s, loc, idx), training=False)
n_keys = 160 - 16 + 1
assert strided_output.shape[1] == len(range(0, n_keys, 3)), strided_output.shape
tf.debugging.assert_near(strided_output[:, :, 0], tf.gather(fast_output[:, :, 0], strided.key_offsets, axis=1))
tf.debugging.assert_near(tf.reduce_sum(strided_output[:, :, 1], axis=1), tf.ones(idx.shape[0]), atol=1e-5)
assert int(strided.key_offsets[-1]) == n_keys - 1
print(f"OK: key_stride=3 uses {strided_output.shape[1]} of {n_keys} keys")

########################################################################################################################
# 3) Training through the Api, saving, loading and embedding with location
########################################################################################################################
with tempfile.TemporaryDirectory() as directory:
    save_path = f"{directory}/fast_ca2e.keras"
    config = make_config("FastCA2E", sites_per_step=2, key_stride=2)
    api = a2e.io.Api(config=config)
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
    tf.debugging.assert_near(loaded((x_s, y_s, loc, idx), training=False), model((x_s, y_s, loc, idx), training=False))

    series = forecasts[:, 0]
    with_loc = api.embed(series, verbose=False, loc=0)
    without_loc = api.embed(series, verbose=False)
    assert with_loc.shape == (n_time - 16 + 1, 8)
    assert float(tf.reduce_max(tf.abs(with_loc - without_loc))) > 0
print("OK: Api.train, save/load and Api.embed(loc=...) work for FastCA2E")
