"""
Randomized PIT (compute_pit) and AnEn in chunks of queries.

Run from the repository root:
    .venv/bin/python -m unittest tests.test_metrics -v
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import unittest
import numpy as np
import a2e


class TestPIT(unittest.TestCase):

    def test_values(self):
        x = np.array([[1.0, 2.0, 2.0, 3.0]] * 4)
        w = np.full((4, 4), 0.25)
        y = np.array([0.5, 2.0, 2.0, 3.5])
        u = np.array([0.3, 0.0, 1.0, 0.7])
        pit = a2e.metrics.computation.compute_pit(x, w, y, u).numpy()
        # below all members: 0; at the tie 2.0: F(y-) = 0.25, F(y) = 0.75; above all members: 1
        np.testing.assert_allclose(pit, [0.0, 0.25, 0.75, 1.0], atol=1e-7)

    def test_weights_and_range(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 100)).astype(np.float32)
        s = rng.normal(scale=5, size=(500, 100)).astype(np.float32)
        w = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
        y = rng.normal(size=500).astype(np.float32)
        pit = a2e.metrics.computation.compute_pit(x, w, y, rng.uniform(size=500)).numpy()
        self.assertTrue(np.all((pit >= 0) & (pit <= 1)))
        # Without ties the PIT is the weighted CDF at y
        np.testing.assert_allclose(pit, (w * (x < y[:, None])).sum(axis=1), atol=1e-5)


class TestAnEnChunks(unittest.TestCase):

    def test_chunk_size_does_not_change_the_ensemble(self):
        rng = np.random.default_rng(0)
        f = rng.normal(5, 2, size=(2400, 2)).astype(np.float32)
        y = rng.normal(5, 2, size=(2400, 1)).astype(np.float32)
        inputs = [f[-152:], f[:2000], y[:2000]]
        ensembles = []
        for chunk_size in [10 ** 9, 64, 7]:
            model = a2e.model.AnEn(n_analogs=50, temporal_window=1, chunk_size=chunk_size)
            model.build(tuple(x.shape for x in inputs))
            ensembles.append(model(inputs).numpy())
        self.assertEqual(ensembles[0].shape, (150, 50))
        for ensemble in ensembles[1:]:
            np.testing.assert_array_equal(ensemble, ensembles[0])


if __name__ == "__main__":
    unittest.main()
