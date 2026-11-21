"""
Training through a2e.io.Api on a small synthetic data set.

- The saved model has the weights of the epoch with the lowest val_loss, also without early stopping (F2).
- Checkpointing is done only by the Api (F2).
- History JSON with ESS, g and the gradient/weight share outside the top-k (F4).
- config.seed fixes the initial weights (F5).

Run from the repository root:
    .venv/bin/python -m unittest tests.test_api_training -v
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import tempfile
import unittest
import numpy as np
import tensorflow as tf
import a2e

EPOCHS = 3
TOP_K = 10


def make_config(**kwargs):
    params = dict(
        model_type="A2E",
        similarity_metric="cosine_similarity",
        d_model=4,
        n_blocks=1,
        seq_len=4,
        lookback=48,
        foresight=1,
        time_to_target=2,
        d_vars=2,
        dropout=0.0,
    )
    params.update(kwargs)
    return a2e.io.ModelConfig(**params)


def synthetic_data(n=260, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 5 + 2 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    x = y + rng.normal(0, 0.8, n)
    return x[:, None].astype(np.float32), y[:, None].astype(np.float32)


def trainable(model):
    """Trainable weights in layer order. model.get_weights() also holds the gradient accumulators of BaseModel,
    which a loaded model does not have yet; layer names carry global counters and differ after loading."""
    return [w.numpy() for w in model.trainable_weights]


class WeightRecorder(tf.keras.callbacks.Callback):
    """Trainable weights at the end of every epoch."""
    def __init__(self):
        super().__init__()
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(trainable(self.model))


class InitialWeights(tf.keras.callbacks.Callback):
    """Trainable weights at the start of training."""
    def on_train_begin(self, logs=None):
        self.weights = trainable(self.model)


class SpoilLastEpoch(tf.keras.callbacks.Callback):
    """
    Sets a very large temperature g right before the validation of the last epoch, so that the last epoch is not
    the best one: the weights collapse onto single members and the SCRPS of the validation explodes.
    """
    def __init__(self, last_epoch):
        super().__init__()
        self.last_epoch = last_epoch
        self.epoch = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_test_begin(self, logs=None):
        if self.epoch == self.last_epoch:
            self.model.cross_attention.g.assign(1e4)


def train(config, save_path, callbacks):
    x, y = synthetic_data()
    features, targets = a2e.io.build_features(x, y, config)
    api = a2e.io.Api(config=config)
    model, history = api.train(
        forecasts=features,
        observations=targets,
        save_path=save_path,
        epochs=EPOCHS,
        test_size=0.3,
        batch_size=16,
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.01),
        loss=a2e.loss.SCRPS(),
        metrics=[a2e.metrics.keras.CRPSMetric(), a2e.metrics.keras.EntropyMetric(), a2e.metrics.keras.ESSMetric()],
        callbacks=callbacks,
        probe_top_k=TOP_K,
    )
    return api, history


class TestBestWeights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.save_path = os.path.join(cls.tmp.name, "A2E_test.keras")
        cls.recorder = WeightRecorder()
        cls.config = make_config()
        cls.api, cls.history = train(cls.config, cls.save_path, [SpoilLastEpoch(EPOCHS - 1), cls.recorder])
        cls.val_loss = cls.history.history["val_loss"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_best_epoch_in_config(self):
        best = int(np.argmin(self.val_loss)) + 1
        self.assertLess(best, EPOCHS, "test setup: the last epoch should not be the best one")
        with open(os.path.splitext(self.save_path)[0] + "_config.json") as f:
            saved = json.load(f)
        self.assertEqual(saved["best_epoch"], best)
        self.assertAlmostEqual(saved["best_val_loss"], float(np.min(self.val_loss)), places=6)
        self.assertEqual(saved["epochs"], EPOCHS)

    def test_saved_model_has_best_weights(self):
        best = int(np.argmin(self.val_loss))
        saved = trainable(tf.keras.models.load_model(self.save_path))
        expected = self.recorder.weights[best]
        self.assertEqual([w.shape for w in saved], [w.shape for w in expected])
        for value, reference in zip(saved, expected):
            np.testing.assert_array_equal(value, reference)
        self.assertTrue(any(not np.array_equal(v, l) for v, l in zip(saved, self.recorder.weights[-1])),
                        "saved weights equal the weights of the last epoch")

    def test_history_json(self):
        with open(a2e.io.Api.history_path(self.save_path)) as f:
            record = json.load(f)
        for key in ["seed", "config", "initial_epoch", "epochs_trained", "best_epoch", "best_val_loss",
                    "stopped_early", "probe", "history"]:
            self.assertIn(key, record)
        self.assertEqual(record["epochs_trained"], EPOCHS)
        self.assertEqual(record["best_epoch"], int(np.argmin(self.val_loss)) + 1)
        self.assertFalse(record["stopped_early"])
        self.assertEqual(record["config"]["seq_len"], self.config.seq_len)
        history = record["history"]
        grad_key, weight_key = f"grad_share_outside_top{TOP_K}", f"weight_share_outside_top{TOP_K}"
        for key in ["loss", "val_loss", "crps", "val_crps", "entropy", "ess", "val_ess", "g", grad_key, weight_key]:
            self.assertEqual(len(history[key]), EPOCHS, key)
        n_archive = self.config.lookback - self.config.seq_len + 1
        for value in history["ess"] + history["val_ess"]:
            self.assertTrue(1.0 <= value <= n_archive)
        for value in history[grad_key] + history[weight_key]:
            self.assertTrue(0.0 <= value <= 1.0)
        # g is the temperature of the cross-attention at the end of each epoch (set to 1e4 in the last one)
        self.assertNotEqual(history["g"][0], history["g"][1])
        self.assertEqual(history["g"][-1], 1e4)

    def test_probe_values(self):
        """ESS on the probe batch is consistent with the weight share outside the top-k."""
        monitor = self.api.monitor
        grad_share, weight_share = monitor.probe()
        z_c, z_h, y_h = self.api.model.encode(monitor.probe_inputs)
        w = tf.nn.softmax(self.api.model.cross_attention.scores(z_c, z_h)).numpy()
        w_sorted = -np.sort(-w, axis=-1)
        np.testing.assert_allclose(weight_share, np.mean(1 - w_sorted[:, :TOP_K].sum(-1)), rtol=1e-5, atol=1e-6)

    def test_euclidean_has_no_g(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            config = make_config(similarity_metric="euclidean_distance")
            train(config, path, [])
            with open(a2e.io.Api.history_path(path)) as f:
                history = json.load(f)["history"]
            self.assertEqual(history["g"], [None] * EPOCHS)

    def test_seed(self):
        initial = {}
        with tempfile.TemporaryDirectory() as tmp:
            for run, seed in [("a", 42), ("b", 42), ("c", 7)]:
                path = os.path.join(tmp, f"{run}.keras")
                recorder = InitialWeights()
                tf.random.uniform(())  # advances the global random state between the runs
                train(make_config(seed=seed), path, [recorder])
                initial[run] = recorder.weights
                with open(a2e.io.Api.history_path(path)) as f:
                    record = json.load(f)
                self.assertEqual(record["seed"], seed)
                self.assertEqual(record["config"]["seed"], seed)
        for a, b in zip(initial["a"], initial["b"]):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(any(not np.array_equal(a, c) for a, c in zip(initial["a"], initial["c"])))

    def test_model_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            with self.assertRaises(ValueError):
                train(make_config(), path, [tf.keras.callbacks.ModelCheckpoint(path)])


if __name__ == "__main__":
    unittest.main()
