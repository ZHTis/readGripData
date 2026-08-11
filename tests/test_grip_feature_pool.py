import unittest

import numpy as np
import pandas as pd

from grip_feature_pool import (
    FEATURE_RECIPES,
    build_grip_feature_pool,
)


class GripFeaturePoolTests(unittest.TestCase):
    def make_split(self):
        sr = 1000.0
        duration_s = 4.0
        n_samples = int(sr * duration_s)
        time_s = np.arange(n_samples) / sr

        # Channel A1 is dominated by 10 Hz; A2 is dominated by 80 Hz.
        signal = np.vstack([
            np.sin(2 * np.pi * 10 * time_s) + 0.1 * np.sin(2 * np.pi * 80 * time_s),
            0.1 * np.sin(2 * np.pi * 10 * time_s) + np.sin(2 * np.pi * 80 * time_s),
        ]).astype(np.float32)
        force = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        phase = np.ones(n_samples, dtype=int)
        phase[500:] = 2
        collision = np.zeros(n_samples, dtype=int)
        collision[3000] = 1

        return {
            "interface_version": "1.0",
            "recording_id": "TEST",
            "segment": "trial",
            "sr": sr,
            "channel_names": ["A1", "A2"],
            "X_list": [signal],
            "target_list": [force[np.newaxis, :]],
            "force_normalized_list": [force],
            "time_list": [time_s],
            "source_time_ms_list": [time_s * 1000.0],
            "state_list": [{"GamePhase": phase, "Collision": collision}],
            "game_phase_labels": {1: "Countdown", 2: "Playing"},
            "meta": pd.DataFrame([{
                "recording_id": "TEST",
                "trial_index0": 0,
                "trial_id": 7,
                "trial_key": "TEST_trial-007",
                "outcome": "failure",
                "collision": True,
                "eeg_start_sample": 10000,
            }]),
        }

    def test_frequency_features_labels_and_sklearn_adapter(self):
        pool = build_grip_feature_pool(
            self.make_split(),
            recipe="wu2022",
            window_ms=500,
            step_ms=100,
            notch_hz=(),
            include_raw_windows=True,
        )
        pool.validate()

        self.assertEqual(pool.X.shape, (36, 2, 5))
        self.assertEqual(pool.raw_windows.shape, (36, 2, 500))
        self.assertEqual(len(pool.labels), len(pool.windows))
        self.assertEqual(pool.windows["trial_id"].unique().tolist(), [7])
        self.assertTrue(pool.windows["mask_flight"].any())
        self.assertEqual(int(pool.labels["collision_onset"].sum()), 1)
        self.assertEqual(pool.labels["outcome"].unique().tolist(), ["failure"])
        self.assertEqual(pool.manifest["recipe"], "wu2022")
        self.assertEqual(pool.manifest["warnings"], [])

        names = pool.feature_names
        ten_hz = names.index("bandpower_4_13Hz")
        eighty_hz = names.index("bandpower_60_150Hz")
        # Ignore early causal-filter transients when comparing spectral power.
        stable = pool.X[10:]
        self.assertGreater(stable[:, 0, ten_hz].mean(), stable[:, 0, eighty_hz].mean())
        self.assertGreater(stable[:, 1, eighty_hz].mean(), stable[:, 1, ten_hz].mean())

        X, y, groups = pool.as_sklearn(
            target="force_normalized",
            mask="mask_flight",
            features=["bandpower_4_13Hz", "bandpower_60_150Hz"],
        )
        self.assertEqual(X.shape[1], 4)  # 2 channels x 2 selected features
        self.assertEqual(len(X), len(y))
        self.assertEqual(set(groups), {"TEST_trial-007"})

    def test_literature_recipe_contains_expected_features(self):
        bands = set(FEATURE_RECIPES["literature_all"]["bands_hz"])
        self.assertIn((0.0, 4.0), bands)
        self.assertIn((70.0, 115.0), bands)
        self.assertIn((60.0, 200.0), bands)
        self.assertTrue(FEATURE_RECIPES["literature_all"]["lmp"])

    def test_non_trial_segment_is_recorded_as_warning(self):
        split = self.make_split()
        split["segment"] = "flight"
        pool = build_grip_feature_pool(
            split,
            recipe="compact",
            window_ms=500,
            step_ms=500,
            notch_hz=(),
        )
        self.assertIn("not segment='trial'", pool.manifest["warnings"][0])


if __name__ == "__main__":
    unittest.main()
