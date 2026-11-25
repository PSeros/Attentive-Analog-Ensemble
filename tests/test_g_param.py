"""
Parametrizations of the temperature g of the cross-attention (ModelConfig.g_param).

- "g": g itself (submitted models). Old models (without g_param in their config) load unchanged, and g is trained
  as before (with weight decay).
- "tau": scores = s / τ, g = 1/τ, τ starts at 1/g_0; τ is excluded from the weight decay of AdamW (Api.train).
- Save and load keep g_param (the weights of a layer are matched by position, not by name).

Run from the repository root:
    .venv/bin/python -m unittest tests.test_g_param -v
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import io
import json
import math
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import a2e
from a2e.layer.cross_attention import CrossAttention, G_PARAMS
from tests.test_api_training import make_config, train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAG = "paper-submission"
SUBMITTED = "experiments/Trained_Models/A2E/Location0/A2E_cosine_similarity_Location0"
N_ARCHIVE = 45  # lookback 48 − seq_len 4 + 1 (make_config)
G_ZERO = math.log2(N_ARCHIVE ** 2 - N_ARCHIVE)
INITIAL_WEIGHT = {"g": G_ZERO, "tau": 1 / G_ZERO}
KERAS_3 = int(tf.keras.__version__.split(".")[0]) >= 3  # the submitted models are Keras 2 models


def attention(g_param, metric="cosine_similarity"):
    layer = CrossAttention(metric, g_zero_init=N_ARCHIVE, g_param=g_param)
    layer.build(((None, 1, 4), (None, None, 4), (None, None, 1)))
    return layer


def build_model(**kwargs):
    config = make_config(**kwargs)
    return a2e.factory.Factory(config).build_model(optimizer="adam", loss=a2e.loss.SCRPS(), metrics=None)


def g_value(model):
    return float(model.cross_attention.g.numpy())


def tag_exists():
    result = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{TAG}"], cwd=PROJECT_ROOT,
                            capture_output=True)
    return result.returncode == 0


def remove_from_saved_config(path, key):
    """Rewrites the config.json in a .keras file without `key` (format of models saved before the key existed)."""
    with zipfile.ZipFile(path) as z:
        files = {name: z.read(name) for name in z.namelist()}
    config = json.loads(files["config.json"])
    del config["config"][key]
    files["config.json"] = json.dumps(config).encode()
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


class TestOldModels(unittest.TestCase):

    def test_config_without_field(self):
        config = make_config().to_dict()
        del config["g_param"]
        self.assertEqual(a2e.io.ModelConfig.from_dict(config).g_param, "g")

    def test_saved_model_without_g_param(self):
        """A model saved before g_param existed (no g_param in its config) loads with g as the weight."""
        tf.keras.utils.set_random_seed(0)
        model = build_model()
        model.cross_attention.g_weight.assign(17.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.keras")
            model.save(path)
            remove_from_saved_config(path, "g_param")
            loaded = tf.keras.models.load_model(path)
        self.assertEqual(loaded.cross_attention.g_param, "g")
        self.assertEqual(g_value(loaded), 17.5)

    @unittest.skipUnless(tag_exists(), f"tag {TAG} not available")
    @unittest.skipIf(KERAS_3, "the submitted Keras 2 models do not load with Keras 3")
    def test_submitted_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = {}
            for suffix in [".keras", "_config.json"]:
                files[suffix] = Path(tmp) / f"model{suffix}"
                with open(files[suffix], "wb") as f:
                    subprocess.run(["git", "show", f"{TAG}:{SUBMITTED}{suffix}"], cwd=PROJECT_ROOT, stdout=f,
                                   check=True)
            with zipfile.ZipFile(files[".keras"]) as z:
                with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as weights:
                    stored_g = float(weights["layers/cross_attention/vars/0"][()])
            config = a2e.io.ModelConfig.load_from_json(files["_config.json"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # receptive field warning for seq_len 80
                model = tf.keras.models.load_model(files[".keras"], compile=False)  # Keras 2 model: no optimizer
        self.assertEqual(config.g_param, "g")
        self.assertEqual(model.cross_attention.g_param, "g")
        self.assertEqual(len(model.cross_attention.weights), 1)
        self.assertEqual(g_value(model), stored_g)


class TestParametrizations(unittest.TestCase):

    def test_initial_values(self):
        """Both start at g = g_0: g = g_0, τ = 1/g_0."""
        for g_param, weight in INITIAL_WEIGHT.items():
            layer = attention(g_param)
            self.assertAlmostEqual(float(layer.g_weight.numpy()), weight, places=5, msg=g_param)
            self.assertAlmostEqual(float(layer.g.numpy()), G_ZERO, places=4, msg=g_param)
            self.assertIn(g_param, layer.g_weight.name)
            self.assertEqual(len(layer.weights), 1)

    def test_scores(self):
        """scores = g · s (tau: s / τ)."""
        rng = np.random.default_rng(0)
        query = tf.constant(rng.normal(size=(2, 1, 4)), tf.float32)
        keys = tf.constant(rng.normal(size=(2, 7, 4)), tf.float32)
        similarity = a2e.metrics.computation.compute_cosine_similarity(query, keys)
        for g_param, weight in [("g", 5.0), ("tau", 0.2)]:
            layer = attention(g_param)
            layer.g_weight.assign(weight)
            self.assertAlmostEqual(float(layer.g.numpy()), 5.0, places=5, msg=g_param)
            np.testing.assert_allclose(layer.scores(query, keys), 5.0 * similarity, rtol=1e-5, err_msg=g_param)

    def test_first_adam_step(self):
        """First Adam step moves the weight by lr against the gradient: g − lr, or 1 / (1/g_0 + lr)."""
        lr = 0.01
        expected = {"g": G_ZERO - lr, "tau": 1 / (1 / G_ZERO + lr)}
        for g_param, value in expected.items():
            layer = attention(g_param)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            with tf.GradientTape() as tape:
                loss = layer.g
            optimizer.apply_gradients([(tape.gradient(loss, layer.g_weight), layer.g_weight)])
            self.assertAlmostEqual(float(layer.g.numpy()), value, places=3, msg=g_param)

    def test_unknown_g_param(self):
        with self.assertRaises(ValueError):
            CrossAttention("cosine_similarity", g_zero_init=N_ARCHIVE, g_param="log_g")

    def test_no_g_for_euclidean(self):
        for g_param in G_PARAMS:
            layer = attention(g_param, metric="euclidean_distance")
            self.assertIsNone(layer.g_weight)
            self.assertFalse(hasattr(layer, "g"))
            self.assertEqual(layer.weights, [])

    def test_config_to_model(self):
        for model_type, extra in [("A2E", {}), ("CA2E", {"max_locations": 3})]:
            for g_param in G_PARAMS:
                model = build_model(model_type=model_type, g_param=g_param, **extra)
                self.assertEqual(model.cross_attention.g_param, g_param, model_type)
                self.assertEqual(model.get_config()["g_param"], g_param, model_type)

    def test_save_and_load(self):
        for g_param, weight in [("g", 21.0), ("tau", 1 / 21.0)]:
            tf.keras.utils.set_random_seed(0)
            model = build_model(g_param=g_param)
            model.cross_attention.g_weight.assign(weight)
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.keras")
                model.save(path)
                loaded = tf.keras.models.load_model(path)
            self.assertEqual(loaded.cross_attention.g_param, g_param)
            self.assertAlmostEqual(g_value(loaded), 21.0, places=3, msg=g_param)


class TestWeightDecay(unittest.TestCase):

    LR, DECAY = 0.1, 0.5

    def decay_step(self, g_param, exclude):
        """One AdamW step with zero gradients: only the weight decay changes the weights."""
        tf.keras.utils.set_random_seed(0)
        api = a2e.io.Api(config=make_config(g_param=g_param))
        api.model = api.factory.build_model(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=self.LR, weight_decay=self.DECAY),
            loss=a2e.loss.SCRPS(), metrics=None)
        if exclude:
            api._exclude_g_from_weight_decay()
        variables = api.model.trainable_variables
        before = [v.numpy() for v in variables]
        api.model.optimizer.apply_gradients([(tf.zeros_like(v), v) for v in variables])
        return api.model, {v.name: (b, v.numpy()) for v, b in zip(variables, before)}

    def assert_kernels_decayed(self, values):
        kernels = [(b, a) for name, (b, a) in values.items() if "kernel" in name]
        self.assertTrue(kernels)
        for b, a in kernels:
            np.testing.assert_allclose(a, b * (1 - self.LR * self.DECAY), rtol=1e-5)

    def test_tau_excluded(self):
        model, values = self.decay_step("tau", exclude=True)
        before, after = values.pop(model.cross_attention.g_weight.name)
        self.assertEqual(after, before)
        self.assert_kernels_decayed(values)

    def test_g_decays_as_before(self):
        """g: the Api changes nothing, g decays like all other weights (as in the submitted models)."""
        model, values = self.decay_step("g", exclude=True)
        before, after = values.pop(model.cross_attention.g_weight.name)
        self.assertAlmostEqual(float(after), float(before) * (1 - self.LR * self.DECAY), places=4)
        self.assert_kernels_decayed(values)

    def test_tau_decays_without_exclusion(self):
        """Control: without the exclusion AdamW decays τ as well."""
        model, values = self.decay_step("tau", exclude=False)
        before, after = values[model.cross_attention.g_weight.name]
        self.assertAlmostEqual(float(after), float(before) * (1 - self.LR * self.DECAY), places=5)

    def test_api_train(self):
        """
        Api.train with tau: τ excluded from the decay, history g = 1/τ, saved model keeps g_param.
        Learning rate 1e-3: τ = 1/g_0 ≈ 0.09 stays far from 0 over the few steps of the test.
        """
        tf.keras.utils.set_random_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "A2E_tau.keras")
            api, _ = train(make_config(g_param="tau"), path, [], learning_rate=1e-3)
            with open(a2e.io.Api.history_path(path)) as f:
                record = json.load(f)
            loaded = tf.keras.models.load_model(path)
        optimizer, attention_layer = api.model.optimizer, api.model.cross_attention
        self.assertFalse(optimizer._use_weight_decay(attention_layer.g_weight))
        kernel = next(v for v in api.model.trainable_variables if "kernel" in v.name)
        self.assertTrue(optimizer._use_weight_decay(kernel))
        self.assertEqual(record["config"]["g_param"], "tau")
        g = record["history"]["g"]
        self.assertTrue(all(G_ZERO / 2 < value < G_ZERO * 2 for value in g), g)
        self.assertNotEqual(g[0], g[-1])
        best = record["best_epoch"] - 1
        self.assertAlmostEqual(g_value(loaded), g[best], places=3)
        self.assertEqual(loaded.cross_attention.g_param, "tau")


if __name__ == "__main__":
    unittest.main()
