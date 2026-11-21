"""
Receptive field of the WaveNet encoders (fix F3).

- Layer count L = ceil(log2(seq_len / n_blocks)) and RF = 1 + n_blocks * (2^L - 1).
- Perturbation test: changing one input time step changes only the embeddings of the RF windows that contain it.
- Exact dependency: the skip sum of the WaveNet blocks at the end of a window has a nonzero gradient exactly on
  the RF time steps of the window. The layers after the blocks (ReLU, dense) act pointwise in time; with random
  weights all ReLU units can be inactive for a window, so the dependency is tested before them. A finite
  perturbation can also change a float32 embedding by less than one ULP at the edge of the receptive field.
- UserWarning if RF != seq_len (80/5), none for the configurations of the retraining (76/5, 3/2).

Run from the repository root:
    .venv/bin/python -m unittest tests.test_receptive_field -v
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import unittest
import warnings
import numpy as np
import tensorflow as tf
from a2e.layer.encoder import Encoder, WaveNet, receptive_field, dilated_layers_per_block
from a2e.layer.spatial_encoder import SpatialEncoder, SpatialWaveNet

LENGTH = 260
STEP = 150  # perturbed time step


def changed_windows(encoder, x, step):
    """Windows (by last time step) whose embedding changes if x[:, step] changes."""
    seq_len = encoder.seq_len
    perturbed = x.copy()
    perturbed[:, step] += 3.0
    z = encoder(tf.constant(x), training=False).numpy()[0]
    z_perturbed = encoder(tf.constant(perturbed), training=False).numpy()[0]
    changed = np.where(np.abs(z - z_perturbed).max(axis=-1) > 0)[0]
    return changed + seq_len - 1  # embedding j <-> window ending at time step j + seq_len - 1


def skip_sum(encoder, x):
    """Sum of the block outputs (skip connections) of the (spatial) WaveNet, before ReLU and the dense layers."""
    h = encoder.projection(x)
    if isinstance(encoder, SpatialEncoder):
        residual, skip = h, None
        blocks = encoder.spatial_wave_net.wave_net_blocks
    else:
        residual, skip = h, tf.zeros_like(h)
        blocks = encoder.wave_net.wave_net_blocks
    for block in blocks:
        residual, skip = block(residual, None, skip, training=False)
    return skip


def dependent_steps(encoder, x, window_end):
    """Input time steps with a nonzero gradient of the skip sum at time step `window_end`."""
    x = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        skip = skip_sum(encoder, x)[0, window_end]
        weights = tf.constant(np.random.default_rng(2).normal(size=skip.shape).astype(np.float32))
        scalar = tf.reduce_sum(skip * weights)
    grad = np.abs(tape.gradient(scalar, x).numpy()[0])
    return np.where(grad.reshape(grad.shape[0], -1).max(axis=-1) > 0)[0]


class TestReceptiveField(unittest.TestCase):

    def setUp(self):
        tf.keras.utils.set_random_seed(0)

    def test_formula(self):
        for seq_len, n_blocks, layers, rf in [(80, 5, 4, 76), (76, 5, 4, 76), (4, 2, 1, 3), (3, 2, 1, 3)]:
            self.assertEqual(dilated_layers_per_block(seq_len, n_blocks), layers)
            self.assertEqual(receptive_field(seq_len, n_blocks), rf)

    def test_layer_count(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seq_len, n_blocks, layers in [(80, 5, 4), (76, 5, 4), (3, 2, 1)]:
                net = WaveNet(d_model=4, seq_len=seq_len, n_blocks=n_blocks)
                self.assertEqual([len(b.dilated_conv.W_f.layers) for b in net.wave_net_blocks], [layers] * n_blocks)

    def check_perturbation(self, seq_len, n_blocks, spatial=False):
        rng = np.random.default_rng(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if spatial:
                encoder = SpatialEncoder(d_model=8, d_loc=3, seq_len=seq_len, n_blocks=n_blocks)
                x = rng.normal(size=(1, LENGTH, 3, 2)).astype(np.float32)
            else:
                encoder = Encoder(d_model=8, seq_len=seq_len, n_blocks=n_blocks)
                x = rng.normal(size=(1, LENGTH, 2)).astype(np.float32)
        rf = receptive_field(seq_len, n_blocks)
        label = f"seq_len={seq_len}, n_blocks={n_blocks}, spatial={spatial}"
        # Only windows that contain the perturbed step change (outside: identical inputs, difference exactly 0)
        changed = changed_windows(encoder, x, STEP)
        self.assertTrue(len(changed) > 0 and set(changed) <= set(range(STEP, STEP + rf)), label)
        # Each window depends on exactly the rf time steps that end with it
        for window_end in [STEP, STEP + rf - 1, LENGTH - 1]:
            np.testing.assert_array_equal(dependent_steps(encoder, x, window_end),
                                          np.arange(window_end - rf + 1, window_end + 1), err_msg=label)

    def test_perturbation_encoder(self):
        for seq_len, n_blocks in [(76, 5), (80, 5), (3, 2)]:
            self.check_perturbation(seq_len, n_blocks)

    def test_perturbation_spatial_encoder(self):
        for seq_len, n_blocks in [(76, 5), (3, 2)]:
            self.check_perturbation(seq_len, n_blocks, spatial=True)

    def test_window_start_discarded(self):
        """With seq_len 80 the first 4 steps of the current window do not enter its embedding."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder = Encoder(d_model=8, seq_len=80, n_blocks=5)
        x = np.random.default_rng(1).normal(size=(1, 80, 2)).astype(np.float32)
        np.testing.assert_array_equal(dependent_steps(encoder, x, window_end=79), np.arange(4, 80))

    def test_warning(self):
        for cls, kwargs in [(WaveNet, {}), (SpatialWaveNet, {"d_loc": 3})]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cls(d_model=4, seq_len=80, n_blocks=5, **kwargs)
            messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
            self.assertTrue(any("receptive field of 1 + 5·(2^4 − 1) = 76" in m and "effective seq_len is 76" in m
                                for m in messages), messages)
            for seq_len, n_blocks in [(76, 5), (3, 2)]:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    cls(d_model=4, seq_len=seq_len, n_blocks=n_blocks, **kwargs)
                self.assertFalse([w for w in caught if "receptive field" in str(w.message)])


if __name__ == "__main__":
    unittest.main()
