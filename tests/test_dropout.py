import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import a2e

# Keras 3 fit() passes training=True only to models whose call() accepts it. Without it, the paper models
# trained without dropout. For every model with dropout: the loss of one fit() step must equal the loss
# without dropout if dropout=0, and must differ from it if dropout>0 (in at least one of three steps, since a
# random mask can leave the loss nearly unchanged).

# Full float32 matmuls on GPUs, so the compiled fit() step and the eager call agree exactly
tf.config.experimental.enable_tensor_float_32_execution(False)

# Small random data: (time, variable) for one location, (time, location, variable) for CA2E/FastCA2E/SA2E
tf.random.set_seed(0)
n_time, n_loc = 600, 3
data = {
    "single": (tf.random.normal((n_time, 2)), tf.random.normal((n_time, 1))),
    "sites": (tf.random.normal((n_time, n_loc, 2)), tf.random.normal((n_time, n_loc, 1))),
    "spatial": (tf.random.normal((n_time, n_loc, 2)), tf.random.normal((n_time, 1))),
}
models = {
    "A2E": ("single", {}),
    "FastA2E": ("single", {}),
    "CA2E": ("sites", {"max_locations": n_loc}),
    "FastCA2E": ("sites", {"max_locations": n_loc, "sites_per_step": 2}),
    "SA2E": ("spatial", {"d_loc": n_loc, "locations_kernel_size": 3}),
}


def losses(model_type, dropout):
    """Loss of one fit() step and loss of the same batch and weights without dropout"""
    kind, kwargs = models[model_type]
    config = a2e.io.ModelConfig(model_type=model_type, similarity_metric="cosine_similarity", d_model=8, n_blocks=2,
                                seq_len=16, lookback=160, foresight=1, time_to_target=2, d_vars=2, dropout=dropout,
                                **kwargs)
    factory = a2e.factory.factory.Factory(config)
    pipeline = factory.build_pipeline(*data[kind], test_size=0.3, batch_size=8)
    model = factory.build_model(optimizer=tf.keras.optimizers.SGD(0.0), loss=a2e.loss.SCRPS(), metrics=None)
    x, y = next(iter(pipeline.get_train_data()))
    without_dropout = float(a2e.loss.SCRPS()(y, model(x, training=False)))
    history = model.fit(tf.data.Dataset.from_tensors((x, y)), epochs=1, verbose=0)
    return float(history.history["loss"][0]), without_dropout


for model_type in models:
    fit_loss, eval_loss = losses(model_type, dropout=0.0)
    np.testing.assert_allclose(fit_loss, eval_loss, rtol=1e-5, err_msg=f"{model_type}: fit() changes the loss without dropout")
    fit_losses = [losses(model_type, dropout=0.5) for _ in range(3)]
    assert any(abs(f - e) > 1e-3 for f, e in fit_losses), f"{model_type}: fit() trains without dropout {fit_losses}"
    print(f"OK: {model_type} fit() uses dropout (dropout 0: {fit_loss:.4f} = {eval_loss:.4f}, "
          f"dropout 0.5: {', '.join(f'{f:.4f} vs {e:.4f}' for f, e in fit_losses)})")
