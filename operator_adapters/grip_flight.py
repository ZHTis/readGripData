from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from align_flight_eeg import align_force_to_eeg, map_task_samples_to_eeg
from read_bci2000 import BCI2000Dat
from seegops import Event, EventTable, Signal, Trial, TrialTable, ValidationError


@dataclass(frozen=True)
class GripFlightRecording:
    recording_id: str
    eeg: Signal
    force: Signal
    trials: TrialTable
    events: EventTable
    eeg_path: Path
    task_path: Path
    available_runs: tuple[str, ...]
    eeg_clock_ms: np.ndarray
    task_clock_ms: np.ndarray
    event_policy: dict[str, Any]

    def seeg_channel_indices(
        self,
        *,
        non_signal_prefixes: tuple[str, ...] = ("EMPTY", "DC"),
        auxiliary_prefixes: tuple[str, ...] = ("EKG", "EMG", "Ear", "Chin", "Eye"),
    ) -> list[int]:
        names = self.eeg.coordinate("channel")
        return [
            index for index, name in enumerate(names)
            if not any(str(name).startswith(prefix) for prefix in non_signal_prefixes)
            and not any(str(name).startswith(prefix) for prefix in auxiliary_prefixes)
        ]


@dataclass(frozen=True)
class GripFlightAdapter:
    """Paradigm-specific composition of generic seegops data contracts."""

    data_dir: Path
    run_name: str
    n_test_trials: int = 1
    force_channel_index: int = 0
    minimum_hold_duration_s: float = 0.25
    minimum_force_change: float = 0.05
    minimum_change_duration_s: float = 0.20
    amplitude_grade_edges: tuple[float, float] | None = None

    @staticmethod
    def find_run_pairs(data_dir: Path) -> list[tuple[str, Path, Path]]:
        pairs = []
        for task_path in sorted(data_dir.rglob("*_1.dat")):
            run_name = task_path.name[:-6]
            eeg_path = task_path.with_name(f"{run_name}.dat")
            if eeg_path.exists():
                pairs.append((run_name, eeg_path, task_path))
        return pairs

    def load(self) -> GripFlightRecording:
        data_dir = Path(self.data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"data directory does not exist: {data_dir}")
        run_pairs = self.find_run_pairs(data_dir)
        selected = [pair for pair in run_pairs if pair[0] == self.run_name]
        if not selected:
            raise FileNotFoundError(
                f"run {self.run_name!r} not found; available: {[row[0] for row in run_pairs]}"
            )
        if len(selected) > 1:
            raise ValidationError(f"run {self.run_name!r} appears more than once: {selected}")
        _, eeg_path, task_path = selected[0]
        eeg_source = BCI2000Dat(eeg_path)
        task_source = BCI2000Dat(task_path)
        required_states = {"SourceTime", "GamePhase"}
        missing = sorted(required_states - set(task_source.state_definitions))
        if missing:
            raise ValidationError(f"task stream is missing states: {missing}")

        eeg_clock_ms, task_clock_ms, aligned_force = align_force_to_eeg(
            eeg_source, task_source
        )
        eeg_time = np.arange(eeg_source.samples) / eeg_source.sampling_rate
        eeg_signal = Signal(
            data=eeg_source.signals.T,
            dims=("channel", "time"),
            coords={
                "channel": np.asarray(eeg_source.channel_names, dtype=object),
                "time": eeg_time,
            },
            sampling_rate=float(eeg_source.sampling_rate),
            unit="uV",
            attrs={
                "recording_id": self.run_name,
                "source_path": str(eeg_path),
                "source_channels": int(eeg_source.source_channels),
                "source_channel_indices": list(range(int(eeg_source.source_channels))),
                "reference": "as_recorded",
            },
            valid_mask=None,
        )
        force_signal = Signal(
            data=np.asarray(aligned_force, dtype=float),
            dims=("time",),
            coords={"time": eeg_time},
            sampling_rate=float(eeg_source.sampling_rate),
            unit="a.u.",
            attrs={
                "recording_id": self.run_name,
                "source_path": str(task_path),
                "source_channel_index": self.force_channel_index,
                "alignment_method": "SourceTime",
            },
            valid_mask=None,
        )
        trials, trial_task_bounds = self._build_trials(
            task_source, eeg_source, eeg_clock_ms, task_clock_ms
        )
        events, event_policy = self._build_events(
            task_source,
            eeg_source,
            eeg_clock_ms,
            task_clock_ms,
            trials,
            trial_task_bounds,
        )
        return GripFlightRecording(
            recording_id=self.run_name,
            eeg=eeg_signal,
            force=force_signal,
            trials=trials,
            events=events,
            eeg_path=eeg_path,
            task_path=task_path,
            available_runs=tuple(row[0] for row in run_pairs),
            eeg_clock_ms=eeg_clock_ms,
            task_clock_ms=task_clock_ms,
            event_policy=event_policy,
        )

    def _build_trials(self, task, eeg, eeg_clock_ms, task_clock_ms):
        phase = task.state("GamePhase").astype(np.int64)
        starts_task = np.flatnonzero((phase == 1) & np.r_[True, phase[:-1] != 1])
        if len(starts_task) <= self.n_test_trials:
            raise ValidationError(
                f"need more than {self.n_test_trials} trials; found {len(starts_task)}"
            )
        stops_task = np.r_[starts_task[1:], task.samples]
        starts_eeg = map_task_samples_to_eeg(starts_task, eeg_clock_ms, task_clock_ms)
        last_eeg = map_task_samples_to_eeg(stops_task - 1, eeg_clock_ms, task_clock_ms)
        stops_eeg = np.minimum(last_eeg + 1, eeg.samples)
        test_from = len(starts_task) - self.n_test_trials
        rows = []
        bounds = {}
        for index, (task_start, task_stop, eeg_start, eeg_stop) in enumerate(
            zip(starts_task, stops_task, starts_eeg, stops_eeg), start=1
        ):
            trial_id = f"{self.run_name}-trial-{index:03d}"
            split = "test" if index - 1 >= test_from else "train"
            rows.append(Trial(
                trial_id=trial_id,
                onset_s=float(eeg_start / eeg.sampling_rate),
                offset_s=float(eeg_stop / eeg.sampling_rate),
                metadata={
                    "recording": self.run_name,
                    "trial_number": index,
                    "split": split,
                    "task_start_sample": int(task_start),
                    "task_stop_sample_exclusive": int(task_stop),
                    "eeg_start_sample": int(eeg_start),
                    "eeg_stop_sample_exclusive": int(eeg_stop),
                },
            ))
            bounds[trial_id] = (int(task_start), int(task_stop))
        return TrialTable(rows), bounds

    @staticmethod
    def _alternating_extrema(force, prominence, distance):
        peaks, _ = scipy_signal.find_peaks(force, prominence=prominence, distance=distance)
        troughs, _ = scipy_signal.find_peaks(-force, prominence=prominence, distance=distance)
        candidates = sorted(
            [(int(index), "peak") for index in peaks]
            + [(int(index), "trough") for index in troughs]
        )
        extrema = []
        for index, kind in candidates:
            if not extrema or extrema[-1][1] != kind:
                extrema.append((index, kind))
                continue
            previous, _ = extrema[-1]
            more_extreme = force[index] > force[previous] if kind == "peak" else force[index] < force[previous]
            if more_extreme:
                extrema[-1] = (index, kind)
        if extrema:
            first, first_kind = extrema[0]
            if abs(force[first] - force[0]) >= prominence:
                extrema.insert(0, (0, "trough" if first_kind == "peak" else "peak"))
            last, last_kind = extrema[-1]
            if abs(force[-1] - force[last]) >= prominence:
                extrema.append((len(force) - 1, "trough" if last_kind == "peak" else "peak"))
        return extrema

    def _build_events(
        self,
        task,
        eeg,
        eeg_clock_ms,
        task_clock_ms,
        trials,
        trial_task_bounds,
    ):
        force = np.asarray(task.signals[:, self.force_channel_index], dtype=float)
        train_trial_ids = [
            row.trial_id for row in trials.trials if row.metadata.get("split") == "train"
        ]
        train_force = np.concatenate([
            force[start:stop] for trial_id, (start, stop) in trial_task_bounds.items()
            if trial_id in train_trial_ids
        ])
        force_mean = float(np.mean(train_force))
        force_sigma = float(np.std(train_force, ddof=0))
        force_range = (force_mean - force_sigma, force_mean + force_sigma)
        min_distance = max(1, round(self.minimum_change_duration_s * task.sampling_rate))
        candidate_rows: list[dict[str, Any]] = []
        for trial in trials.trials:
            task_start, task_stop = trial_task_bounds[trial.trial_id]
            trial_force = force[task_start:task_stop]
            extrema = self._alternating_extrema(
                trial_force, self.minimum_force_change, min_distance
            )
            for (local_start, _), (local_stop, _) in zip(extrema[:-1], extrema[1:]):
                start = task_start + local_start
                stop = task_start + local_stop
                delta = float(force[stop] - force[start])
                duration_s = (stop - start) / task.sampling_rate
                if abs(delta) < self.minimum_force_change or duration_s < self.minimum_change_duration_s:
                    continue
                candidate_rows.append({
                    "event_type": "exertion" if delta > 0 else "relaxation",
                    "trial_id": trial.trial_id,
                    "split": trial.metadata.get("split"),
                    "task_start": start,
                    "task_stop": stop,
                    "start_force": float(force[start]),
                    "end_force": float(force[stop]),
                    "delta_force": delta,
                    "amplitude": abs(delta),
                    "mean_force": float(np.mean(force[start:stop + 1])),
                })

            in_range = (trial_force >= force_range[0]) & (trial_force <= force_range[1])
            edges = np.diff(np.r_[False, in_range, False].astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)
            minimum_samples = max(1, round(self.minimum_hold_duration_s * task.sampling_rate))
            for local_start, local_stop in zip(starts, stops):
                if local_stop - local_start < minimum_samples:
                    continue
                start = task_start + int(local_start)
                stop = task_start + int(local_stop - 1)
                values = force[start:stop + 1]
                candidate_rows.append({
                    "event_type": "hold",
                    "trial_id": trial.trial_id,
                    "split": trial.metadata.get("split"),
                    "task_start": start,
                    "task_stop": stop,
                    "start_force": float(values[0]),
                    "end_force": float(values[-1]),
                    "delta_force": float(values[-1] - values[0]),
                    "amplitude": float(np.ptp(values)),
                    "mean_force": float(np.mean(values)),
                })

        training_exertion_amplitudes = np.asarray([
            row["amplitude"] for row in candidate_rows
            if row["split"] == "train" and row["event_type"] == "exertion"
        ])
        if self.amplitude_grade_edges is None:
            if len(training_exertion_amplitudes) < 3:
                raise ValidationError("too few training exertion events to fit grade tertiles")
            grade_edges = tuple(
                float(value) for value in np.quantile(training_exertion_amplitudes, [1 / 3, 2 / 3])
            )
            grade_source = "training exertion amplitude tertiles"
        else:
            grade_edges = tuple(float(value) for value in self.amplitude_grade_edges)
            grade_source = "fixed user thresholds"
        if len(grade_edges) != 2 or not grade_edges[0] < grade_edges[1]:
            raise ValidationError("amplitude grade edges must be increasing")

        candidate_rows.sort(key=lambda row: (row["task_start"], row["event_type"]))
        event_rows = []
        for number, row in enumerate(candidate_rows, start=1):
            start_eeg = int(map_task_samples_to_eeg(
                np.asarray([row["task_start"]]), eeg_clock_ms, task_clock_ms
            )[0])
            stop_eeg = int(map_task_samples_to_eeg(
                np.asarray([row["task_stop"]]), eeg_clock_ms, task_clock_ms
            )[0])
            amplitude = row["amplitude"]
            if row["event_type"] == "hold":
                grade = "in_range"
                label = "hold_in_range"
            else:
                grade = (
                    "small" if amplitude <= grade_edges[0]
                    else "medium" if amplitude <= grade_edges[1]
                    else "large"
                )
                label = f"{row['event_type']}_{grade}"
            duration_s = max(0.0, (stop_eeg - start_eeg) / eeg.sampling_rate)
            event_rows.append(Event(
                event_id=f"{self.run_name}-force-{number:05d}",
                event_type=row["event_type"],
                onset_s=float(start_eeg / eeg.sampling_rate),
                duration_s=float(duration_s),
                trial_id=row["trial_id"],
                value=float(amplitude),
                metadata={
                    "split": row["split"],
                    "label": label,
                    "grade": grade,
                    "task_start_sample": row["task_start"],
                    "task_stop_sample": row["task_stop"],
                    "eeg_start_sample": start_eeg,
                    "eeg_stop_sample": stop_eeg,
                    "start_force": row["start_force"],
                    "end_force": row["end_force"],
                    "delta_force": row["delta_force"],
                    "amplitude": amplitude,
                    "mean_force": row["mean_force"],
                },
            ))
        policy = {
            "force_range": force_range,
            "force_mean_train": force_mean,
            "force_sigma_train": force_sigma,
            "minimum_hold_duration_s": self.minimum_hold_duration_s,
            "minimum_force_change": self.minimum_force_change,
            "minimum_change_duration_s": self.minimum_change_duration_s,
            "amplitude_grade_edges": grade_edges,
            "grade_source": grade_source,
        }
        return EventTable(event_rows, trials), policy
