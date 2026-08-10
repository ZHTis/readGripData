from pathlib import Path
import unittest

import numpy as np

from grip_data_interface import build_flight_trial_interface, select_flight_split


class FakeRecording:
    def __init__(self, path, signals, sampling_rate, channel_names, states):
        self.path = Path(path)
        self.signals = np.asarray(signals)
        self.samples = self.signals.shape[0]
        self.source_channels = self.signals.shape[1]
        self.sampling_rate = float(sampling_rate)
        self.channel_names = list(channel_names)
        self._states = {name: np.asarray(values) for name, values in states.items()}
        self.state_definitions = {name: (16, 0, 0) for name in states}

    def state(self, name):
        return self._states[name]


class GripDataInterfaceTests(unittest.TestCase):
    def make_recordings(self):
        eeg_n = 1000
        task_n = 100
        eeg = FakeRecording(
            "eeg.dat",
            np.column_stack([np.arange(eeg_n), np.full(eeg_n, -1)]),
            1000,
            ["A1", "EMPTY1"],
            {"SourceTime": np.arange(eeg_n)},
        )
        phase = np.zeros(task_n, dtype=int)
        phase[10:20] = 1
        phase[20:50] = 2
        phase[50] = 3
        phase[51:60] = 5
        phase[60:70] = 1
        phase[70:95] = 2
        phase[95:] = 4
        collision = np.zeros(task_n, dtype=int)
        collision[50] = 1
        result = np.zeros(task_n, dtype=int)
        result[51:60] = 2
        result[95:] = 1
        collision_object = np.zeros(task_n, dtype=int)
        collision_object[50] = 3
        normalized = np.linspace(0, 65535, task_n).astype(int)
        task = FakeRecording(
            "task.dat",
            np.linspace(0.8, 1.2, task_n)[:, None],
            100,
            ["Grip"],
            {
                "SourceTime": np.arange(task_n) * 10,
                "GamePhase": phase,
                "Collision": collision,
                "CollisionObject": collision_object,
                "FlightTrialResult": result,
                "GripForceRaw": (np.linspace(0.8, 1.2, task_n) * 10000).astype(int),
                "GripForceNormalized": normalized,
                "BallWorldX": np.arange(task_n) * 100,
                "BallWorldY": np.full(task_n, 5000),
                "BallVelocityY": np.full(task_n, 32768),
                "CameraWorldX": np.arange(task_n) * 100,
            },
        )
        return eeg, task

    def test_variable_length_interface_and_semantic_events(self):
        eeg, task = self.make_recordings()
        data = build_flight_trial_interface(eeg, task)
        self.assertEqual(data["interface_version"], "1.0")
        self.assertEqual(data["recording_id"], "eeg")
        self.assertEqual(data["n_ch"], 1)
        self.assertEqual(data["channel_names"], ["A1"])
        self.assertEqual(len(data["X_list"]), 2)
        self.assertEqual(data["X_list"][0].shape[0], 1)
        self.assertEqual(data["target_list"][0].shape, (1, data["X_list"][0].shape[1]))
        self.assertEqual(data["meta"]["outcome"].tolist(), ["failure", "success"])
        self.assertTrue(bool(data["meta"].loc[0, "collision"]))
        self.assertNotIn("code", data["events"].columns)
        collision = data["events"].loc[data["events"]["event"] == "collision_onset"]
        self.assertEqual(collision.iloc[0]["label"], "object_3")

    def test_outcome_splits_preserve_trial_ids(self):
        eeg, task = self.make_recordings()
        data = build_flight_trial_interface(eeg, task)
        success = select_flight_split(data, "success")
        failure = select_flight_split(data, "failure")
        self.assertEqual(success["meta"]["trial_id"].tolist(), [2])
        self.assertEqual(failure["meta"]["trial_id"].tolist(), [1])
        self.assertEqual(len(success["X_list"]), 1)
        self.assertEqual(len(failure["event_list"]), 1)


if __name__ == "__main__":
    unittest.main()
