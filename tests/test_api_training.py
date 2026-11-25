"""
Training through a2e.io.Api on a small synthetic data set.

- The saved model has the weights of the epoch with the lowest val_loss, also without early stopping (F2).
- Checkpointing is done only by the Api (F2).

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


class SpoilLastEpoch(tf.keras.callbacks.Callback):
    """Randomizes the weights at the start of the last epoch, so that the last epoch is not the best one."""
    def __init__(self, last_epoch):
        super().__init__()
        self.last_epoch = last_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.last_epoch:
            rng = np.random.default_rng(1)
            self.model.set_weights([w + rng.normal(0, 3, w.shape).astype(w.dtype)
                                    if w.dtype.kind == "f" else w for w in self.model.get_weights()])


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
        metrics=[a2e.metrics.keras.CRPSMetric(), a2e.metrics.keras.EntropyMetric()],
        callbacks=callbacks,
    )
    return api, history


class TestBestWeights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

    def test_model_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            with self.assertRaises(ValueError):
                train(make_config(), path, [tf.keras.callbacks.ModelCheckpoint(path)])


if __name__ == "__main__":
    unittest.main()
