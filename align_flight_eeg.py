"""Align a BCI2000 flight-task recording to its EEG recording by SourceTime."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from read_bci2000 import BCI2000Dat


def source_clock_ms(recording: BCI2000Dat) -> np.ndarray:
    """Return a continuous per-sample clock interpolated from SourceTime blocks."""
    raw = recording.state("SourceTime").astype(np.int64)
    changes = np.flatnonzero(np.r_[True, np.diff(raw) != 0])
    anchor_values = raw[changes].copy()
    anchor_values += np.r_[0, np.cumsum(np.diff(anchor_values) < -30000) * 65536]
    return np.interp(np.arange(recording.samples), changes, anchor_values)


def put_clocks_on_same_wrap(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Shift the secondary uint16 clock by whole wraps to match the primary."""
    wrap_shift = round((primary[0] - secondary[0]) / 65536)
    return secondary + wrap_shift * 65536


def align_force_to_eeg(
    eeg: BCI2000Dat, task: BCI2000Dat
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return EEG clock, task clock, and force resampled onto the EEG clock."""
    eeg_clock = source_clock_ms(eeg)
    task_clock = put_clocks_on_same_wrap(eeg_clock, source_clock_ms(task))
    force_at_eeg_rate = np.interp(
        eeg_clock,
        task_clock,
        np.asarray(task.signals[:, 0], dtype=float),
        left=np.nan,
        right=np.nan,
    )
    return eeg_clock, task_clock, force_at_eeg_rate


def map_task_samples_to_eeg(
    task_samples: np.ndarray, eeg_clock: np.ndarray, task_clock: np.ndarray
) -> np.ndarray:
    """Map task sample indices to nearest EEG sample indices."""
    insertion = np.searchsorted(eeg_clock, task_clock[task_samples])
    insertion = np.clip(insertion, 1, len(eeg_clock) - 1)
    before = insertion - 1
    use_before = (
        np.abs(eeg_clock[before] - task_clock[task_samples])
        <= np.abs(eeg_clock[insertion] - task_clock[task_samples])
    )
    return np.where(use_before, before, insertion)


def flight_event_table(
    eeg: BCI2000Dat, task: BCI2000Dat, eeg_clock: np.ndarray, task_clock: np.ndarray
) -> list[dict[str, object]]:
    phase = task.state("GamePhase").astype(int)
    collision = task.state("Collision").astype(int)
    result = task.state("FlightTrialResult").astype(int)
    trial_starts = np.flatnonzero((phase == 1) & np.r_[True, phase[:-1] != 1])

    trial_at_sample = np.zeros(task.samples, dtype=int)
    for number, start in enumerate(trial_starts, 1):
        end = trial_starts[number] if number < len(trial_starts) else task.samples
        trial_at_sample[start:end] = number

    event_specs = [
        ("trial_start", np.flatnonzero((phase == 1) & np.r_[True, phase[:-1] != 1]), phase),
        ("flight_start", np.flatnonzero((phase == 2) & np.r_[True, phase[:-1] != 2]), phase),
        ("collision_onset", np.flatnonzero((collision > 0) & np.r_[True, collision[:-1] == 0]), collision),
        ("result_onset", np.flatnonzero((result > 0) & np.r_[True, result[:-1] == 0]), result),
    ]
    rows = []
    for event_type, task_indices, values in event_specs:
        eeg_indices = map_task_samples_to_eeg(task_indices, eeg_clock, task_clock)
        for task_index, eeg_index in zip(task_indices, eeg_indices):
            rows.append({
                "trial": int(trial_at_sample[task_index]),
                "event": event_type,
                "code": int(values[task_index]),
                "task_sample": int(task_index),
                "task_time_s": float(task_index / task.sampling_rate),
                "source_clock_ms": float(task_clock[task_index]),
                "eeg_sample": int(eeg_index),
                "eeg_time_s": float(eeg_index / eeg.sampling_rate),
                "alignment_error_ms": float(eeg_clock[eeg_index] - task_clock[task_index]),
            })
    return sorted(rows, key=lambda row: int(row["task_sample"]))


def export_events(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eeg_file", type=Path)
    parser.add_argument("task_file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    eeg = BCI2000Dat(args.eeg_file)
    task = BCI2000Dat(args.task_file)
    eeg_clock, task_clock, force_at_eeg_rate = align_force_to_eeg(eeg, task)
    rows = flight_event_table(eeg, task, eeg_clock, task_clock)
    output = args.output or args.task_file.with_name(
        f"{args.task_file.stem}_eeg_aligned_events.csv"
    )
    export_events(rows, output)
    errors = np.array([row["alignment_error_ms"] for row in rows], dtype=float)
    print(f"Wrote {len(rows)} events to {output}")
    print(f"Force samples aligned to EEG grid: {np.count_nonzero(~np.isnan(force_at_eeg_rate))}")
    print(f"Maximum absolute event alignment error: {np.max(np.abs(errors)):.3f} ms")


if __name__ == "__main__":
    main()
