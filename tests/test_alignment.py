"""
Alignment of queries, targets and archive values (fix F1).

Synthetic data with the time index as value: x_tau = tau, y_tau = tau + 0.5. Every value in a sample can then be
decoded into the time it belongs to. Conditions, for the training pipelines and the evaluation helpers:
- target time = time of the last forecast in the query window - foresight
- newest observation in the query window = target time - time_to_target (issue time)
- every archive window carries the value at its own target time (last forecast time - foresight)
- no archive value lies after the issue time

Run from the repository root:
    .venv/bin/python -m unittest tests.test_alignment -v
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import unittest
import numpy as np
import tensorflow as tf
import a2e

T_TOTAL = 220
LOOKBACK = 40
SEQ_LEN = 5
FORESIGHT = 1
TIME_TO_TARGET = 2
LOC_OFFSET = 1000.0  # value offset per location, decoded as tau = value - LOC_OFFSET * location


def make_config(model_type, **kwargs):
    return a2e.io.ModelConfig(
        model_type=model_type,
        similarity_metric="cosine_similarity",
        d_model=4,
        n_blocks=1,
        seq_len=SEQ_LEN,
        lookback=LOOKBACK,
        foresight=FORESIGHT,
        time_to_target=TIME_TO_TARGET,
        d_vars=2,
        **kwargs,
    )


def raw_data(n_locations=None):
    """x_tau = tau, y_tau = tau + 0.5 (+ LOC_OFFSET * location)."""
    tau = np.arange(T_TOTAL, dtype=np.float32)
    if n_locations is None:
        return tau[:, None], tau[:, None] + 0.5
    offset = LOC_OFFSET * np.arange(n_locations, dtype=np.float32)
    x = tau[:, None] + offset[None, :]
    return x[..., None], x[..., None] + 0.5


def sample(dataset, i):
    return next(iter(dataset.skip(i).take(1)))


class AlignmentChecks:
    """Assertions on decoded times. `fc` and `obs` are times (not values)."""

    def check_query(self, fc_last, obs_last, target):
        self.assertEqual(target, fc_last - FORESIGHT, "target time != last forecast time - foresight")
        self.assertEqual(obs_last, target - TIME_TO_TARGET, "newest observation in query != issue time")

    def check_archive(self, fc_last, obs_last, members, issue_time):
        np.testing.assert_array_equal(members, fc_last - FORESIGHT,
                                      err_msg="archive value != value at the window's target time")
        np.testing.assert_array_equal(obs_last, members - TIME_TO_TARGET,
                                      err_msg="newest observation in archive window != its issue time")
        self.assertLessEqual(members.max(), issue_time, "archive value after the issue time")


class TestPipelineAlignment(AlignmentChecks, unittest.TestCase):

    def check_sample(self, x_t, x_h, y_h, y_t):
        """x_* in values of feature rows [x, y]; returns nothing, asserts the conditions."""
        x_t, x_h, y_h, y_t = (np.asarray(v, dtype=np.float64) for v in (x_t, x_h, y_h, y_t))
        self.assertEqual(x_t.shape[0], SEQ_LEN)
        target = float(y_t.squeeze()) - 0.5
        self.check_query(fc_last=x_t[-1, 0], obs_last=x_t[-1, 1] - 0.5, target=target)
        # The model uses keys of the windows ending at positions seq_len-1 ... lookback-1 and y_h[:, seq_len-1:]
        keys = slice(SEQ_LEN - 1, None)
        self.check_archive(fc_last=x_h[keys, 0], obs_last=x_h[keys, 1] - 0.5,
                           members=y_h[keys, 0] - 0.5, issue_time=target - TIME_TO_TARGET)

    def test_a2e_pipeline(self):
        config = make_config("A2E")
        x, y = raw_data()
        features, targets = a2e.io.build_features(x, y, config)
        pipeline = a2e.pipeline.A2EPipeline(features, targets, config, test_size=0.3, batch_size=4)
        dataset = pipeline._create_dataset(features, targets)
        for i in (0, pipeline.num_samples // 2, pipeline.num_samples - 1):
            (x_t, x_h, y_h), y_t = sample(dataset, i)
            self.check_sample(x_t, x_h, y_h, y_t)

    def test_ca2e_pipeline(self):
        n_locations = 3
        config = make_config("CA2E", max_locations=n_locations)
        x, y = raw_data(n_locations)
        features, targets = a2e.io.build_features(x, y, config)
        pipeline = a2e.pipeline.CA2EPipeline(features, targets, config, test_size=0.3, batch_size=4)
        dataset = pipeline._create_dataset(features, targets)
        for i in (0, 1, 2, pipeline.num_samples // 2, pipeline.num_samples - 1):
            (x_t, x_h, y_h, loc), y_t = sample(dataset, i)
            offset = LOC_OFFSET * int(loc)
            self.check_sample(np.asarray(x_t) - offset, np.asarray(x_h) - offset,
                              np.asarray(y_h) - offset, np.asarray(y_t) - offset)

    def test_sa2e_pipeline(self):
        d_loc = 3
        config = make_config("SA2E", d_loc=d_loc)
        x, y = raw_data(d_loc)
        features, targets = a2e.io.build_features(x, y, config, target_location=0)
        self.assertEqual(tuple(targets.shape), (T_TOTAL - FORESIGHT - TIME_TO_TARGET, 1))
        pipeline = a2e.pipeline.SA2EPipeline(features, targets, config, test_size=0.3, batch_size=4)
        dataset = pipeline._create_dataset(features, targets)
        for i in (0, pipeline.num_samples // 2, pipeline.num_samples - 1):
            (x_t, x_h, y_h), y_t = sample(dataset, i)
            # Target location 0 has no offset; the neighbours carry the same times with an offset.
            self.check_sample(np.asarray(x_t)[:, 0], np.asarray(x_h)[:, 0], y_h, y_t)
            np.testing.assert_array_equal(np.asarray(x_h)[:, 2] - 2 * LOC_OFFSET, np.asarray(x_h)[:, 0])


class TestEvaluationSplit(AlignmentChecks, unittest.TestCase):
    test_size = 30

    def setUp(self):
        self.config = make_config("A2E")
        x, y = raw_data()
        self.features, self.targets = a2e.io.build_features(x, y, self.config)
        self.split = a2e.io.evaluation_split(self.features, self.targets, self.config, self.test_size)
        # Stand-in for Api.embed: embedding j belongs to the window that ends in row j + seq_len - 1
        self.embeddings = np.asarray(self.features)[SEQ_LEN - 1:]

    def test_build_features_convention(self):
        x, y = raw_data()
        idx = FORESIGHT + TIME_TO_TARGET
        np.testing.assert_array_equal(self.features, np.concatenate([x[idx:], y[:-idx]], axis=-1))
        np.testing.assert_array_equal(self.targets, y[idx:])

    def test_build_features_variants(self):
        x, y = raw_data(3)
        config = make_config("A2E")
        features, targets = a2e.io.build_features(x, y, config, target_location=1)
        self.assertEqual(tuple(features.shape), (T_TOTAL - 3, 3, 2))
        np.testing.assert_array_equal(targets, y[3:, 1])
        no_obs = make_config("A2E")
        no_obs.d_vars = 1
        features, _ = a2e.io.build_features(*raw_data(), no_obs, observation_inputs=False)
        np.testing.assert_array_equal(features, raw_data()[0][3:])
        with self.assertRaises(ValueError):
            a2e.io.build_features(*raw_data(), no_obs)  # d_vars mismatch

    def test_queries(self):
        query = self.embeddings[self.split.query]
        targets = np.asarray(self.split.y_t, dtype=np.float64) - 0.5
        self.assertEqual(len(targets), self.test_size)
        self.assertEqual(len(query), self.test_size)
        for fc_last, obs_last, target in zip(query[:, 0], query[:, 1] - 0.5, targets):
            self.check_query(fc_last, obs_last, target)
        # Raw time index of the targets; the last window ends with the last forecast x_{T-1} = x_{t+foresight}
        np.testing.assert_array_equal(self.split.target_index, targets)
        self.assertEqual(self.split.target_index[-1], T_TOTAL - 1 - FORESIGHT)

    def test_archive(self):
        query = self.embeddings[self.split.query]
        archive = self.embeddings[self.split.archive]
        members = np.asarray(self.split.y_h, dtype=np.float64) - 0.5
        self.assertEqual(len(archive), len(members))
        first_issue_time = query[0, 1] - 0.5
        self.check_archive(fc_last=archive[:, 0], obs_last=archive[:, 1] - 0.5,
                           members=members, issue_time=first_issue_time)
        # The archive is cut only as far as needed: the newest value is the one at the issue time
        self.assertEqual(members.max(), first_issue_time)
        # The archive starts with the first complete window
        self.assertEqual(self.split.archive.start, 0)

    def test_split_with_end(self):
        """Queries that end before `end` (training period), archive before them."""
        end = len(self.features) - self.test_size
        split = a2e.io.evaluation_split(self.features, self.targets, self.config, test_size=50, end=end)
        query = self.embeddings[split.query]
        archive = self.embeddings[split.archive]
        self.assertEqual(split.first_query_row + 50, end)
        targets = np.asarray(split.y_t, dtype=np.float64) - 0.5
        for fc_last, obs_last, target in zip(query[:, 0], query[:, 1] - 0.5, targets):
            self.check_query(fc_last, obs_last, target)
        self.check_archive(archive[:, 0], archive[:, 1] - 0.5, np.asarray(split.y_h) - 0.5, query[0, 1] - 0.5)

    def test_anen_inputs(self):
        window = FORESIGHT
        current, archive_f, archive_y = a2e.io.anen_inputs(self.features, self.targets, self.split, window)
        width = 2 * window + 1
        frames = tf.signal.frame(current, width, 1, axis=0).numpy()
        self.assertEqual(len(frames), self.test_size)
        targets = np.asarray(self.split.y_t, dtype=np.float64) - 0.5
        # Same query windows as A2E: last forecast in the frame = target time + foresight
        np.testing.assert_array_equal(frames[:, -1, 0], targets + FORESIGHT)
        # Archive: AnEn takes the target of the frame center, archive_y[window:-window]
        archive_frames = tf.signal.frame(archive_f, width, 1, axis=0).numpy()
        members = np.asarray(archive_y)[window:-window, 0] - 0.5
        self.check_archive(archive_frames[:, -1, 0], archive_frames[:, -1, 1] - 0.5, members,
                           issue_time=targets[0] - TIME_TO_TARGET)
        self.assertEqual(self.split.last_archive_row, len(archive_f) - 1)


if __name__ == "__main__":
    unittest.main()
