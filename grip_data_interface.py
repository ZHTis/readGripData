"""Team-facing interface for aligned flight-task EEG and grip-force trials.

The returned ``flight_split`` follows the variable-length trial convention used
by ``playgroundgit``: ``X_list`` stores one channels-by-samples EEG array per
trial, continuous targets are stored in parallel lists, and ``meta`` is a
row-per-trial pandas DataFrame.  Raw recordings remain outside version control.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from align_flight_eeg import align_force_to_eeg, flight_event_table, map_task_samples_to_eeg
from read_bci2000 import BCI2000Dat


INTERFACE_VERSION = "1.0"
SPLIT_NAMES = ("all", "success", "failure", "collision")
SEGMENT_NAMES = ("flight", "playing", "trial")

# GripFlightTask GamePhase state values (see GameStateMachine::State):
#   0 Idle             - run has not entered a trial yet
#   1 Countdown        - trial preparation / pre-feedback period
#   2 Playing          - active flight; grip force controls vertical acceleration
#   3 Hit              - a collision was detected in the active feedback block
#   4 TrialSuccess     - success/result display after completing the flight
#   5 TrialFailure     - failure/result display after a collision
#   6 InterTrial       - interval between two trials
#   7 SessionComplete  - the run has stopped or all trials are complete
#
# Segment behavior in this interface:
#   flight  -> starts at the first Playing sample; ends before result display,
#              while retaining the Hit/Collision onset sample for failures
#   playing -> keeps only the contiguous GamePhase == 2 interval
#   trial   -> starts at Countdown and continues until the next Countdown
GAME_PHASE_LABELS = {
    0: "Idle",
    1: "Countdown",
    2: "Playing",
    3: "Hit",
    4: "TrialSuccess",
    5: "TrialFailure",
    6: "InterTrial",
    7: "SessionComplete",
}

_CONTINUOUS_STATES = {
    "GripForceRaw": (1.0 / 10000.0, 0.0),
    "GripForceNormalized": (1.0 / 65535.0, 0.0),
    "BallWorldX": (1.0 / 100.0, 0.0),
    "BallWorldY": (1.0 / 100.0, 0.0),
    "BallVelocityY": (1.0 / 100.0, -327.68),
    "CameraWorldX": (1.0 / 100.0, 0.0),
}
_DISCRETE_STATES = (
    "GamePhase",
    "Collision",
    "CollisionObject",
    "FlightTrialResult",
)
_TRIAL_LIST_KEYS = (
    "X_list",
    "target_list",
    "force_list",
    "force_normalized_list",
    "time_list",
    "source_time_ms_list",
    "state_list",
    "event_list",
)
_EVENT_COLUMNS = (
    "trial_id",
    "event",
    "label",
    "task_sample",
    "task_time_s",
    "source_clock_ms",
    "eeg_sample",
    "eeg_time_s",
    "alignment_error_ms",
)


def _require_states(recording: BCI2000Dat, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in recording.state_definitions]
    if missing:
        raise KeyError(f"Missing required BCI2000 states in {recording.path}: {missing}")


def _resolve_channels(
    eeg: BCI2000Dat,
    channel_indices: Iterable[int] | None,
    drop_empty_channels: bool,
) -> tuple[np.ndarray, list[str]]:
    if channel_indices is None:
        if drop_empty_channels:
            indices = np.asarray(
                [i for i, name in enumerate(eeg.channel_names) if not name.upper().startswith("EMPTY")],
                dtype=int,
            )
        else:
            indices = np.arange(eeg.source_channels, dtype=int)
    else:
        indices = np.asarray(list(channel_indices), dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("channel_indices must select at least one channel")
    if np.any(indices < 0) or np.any(indices >= eeg.source_channels):
        raise IndexError(f"channel_indices must be within 0..{eeg.source_channels - 1}")
    return indices, [eeg.channel_names[i] for i in indices]


def _nearest_task_values(
    values: np.ndarray, query_clock: np.ndarray, task_clock: np.ndarray
) -> np.ndarray:
    insertion = np.searchsorted(task_clock, query_clock)
    insertion = np.clip(insertion, 1, len(task_clock) - 1)
    before = insertion - 1
    choose_before = (
        np.abs(query_clock - task_clock[before])
        <= np.abs(task_clock[insertion] - query_clock)
    )
    nearest = np.where(choose_before, before, insertion)
    return np.asarray(values)[nearest]


def _task_trial_bounds(task: BCI2000Dat, segment: str) -> list[dict[str, Any]]:
    """Find trial/flight bounds from GamePhase transitions.

    Countdown (1) marks a new trial and Playing (2) marks active control.  For
    failed flights, Hit (3) and the Collision onset are retained by the default
    ``flight`` segment, while TrialFailure (5) result-display samples are not.
    """
    if segment not in SEGMENT_NAMES:
        raise ValueError(f"segment must be one of {SEGMENT_NAMES}, got {segment!r}")
    _require_states(task, ("GamePhase", "Collision", "FlightTrialResult"))
    phase = task.state("GamePhase").astype(int)
    collision = task.state("Collision").astype(int)
    result = task.state("FlightTrialResult").astype(int)
    trial_starts = np.flatnonzero((phase == 1) & np.r_[True, phase[:-1] != 1])
    if trial_starts.size == 0:
        raise ValueError(f"No Countdown trial onsets found in {task.path}")

    bounds: list[dict[str, Any]] = []
    for trial_index0, trial_start in enumerate(trial_starts):
        trial_stop = (
            int(trial_starts[trial_index0 + 1])
            if trial_index0 + 1 < len(trial_starts)
            else task.samples
        )
        trial_indices = np.arange(int(trial_start), int(trial_stop))
        playing = trial_indices[phase[trial_indices] == 2]
        if playing.size == 0:
            continue
        collision_indices = trial_indices[collision[trial_indices] > 0]
        result_indices = trial_indices[result[trial_indices] > 0]

        if segment == "trial":
            start, stop = int(trial_start), int(trial_stop)
        elif segment == "playing":
            start, stop = int(playing[0]), int(playing[-1] + 1)
        else:
            start = int(playing[0])
            if collision_indices.size:
                stop = int(collision_indices[0] + 1)
            elif result_indices.size:
                stop = int(result_indices[0])
            else:
                stop = int(playing[-1] + 1)

        if stop <= start:
            continue
        result_value = int(result[trial_indices].max(initial=0))
        outcome = {1: "success", 2: "failure"}.get(result_value, "incomplete")
        bounds.append({
            "trial_index0": int(trial_index0),
            "trial_id": int(trial_index0 + 1),
            "task_trial_start": int(trial_start),
            "task_trial_stop": int(trial_stop),
            "task_start": start,
            "task_stop": stop,
            "flight_start": int(playing[0]),
            "collision_sample": int(collision_indices[0]) if collision_indices.size else None,
            "result_sample": int(result_indices[0]) if result_indices.size else None,
            "outcome": outcome,
            "collision": bool(collision_indices.size),
        })
    return bounds


def _semantic_events(
    eeg: BCI2000Dat,
    task: BCI2000Dat,
    eeg_clock: np.ndarray,
    task_clock: np.ndarray,
) -> pd.DataFrame:
    rows = flight_event_table(eeg, task, eeg_clock, task_clock)
    collision_object = (
        task.state("CollisionObject").astype(int)
        if "CollisionObject" in task.state_definitions
        else None
    )
    semantic_rows = []
    for row in rows:
        event = str(row["event"])
        task_sample = int(row["task_sample"])
        if event == "trial_start":
            label = "Countdown"
        elif event == "flight_start":
            label = "Playing"
        elif event == "collision_onset":
            obj = int(collision_object[task_sample]) if collision_object is not None else 0
            label = f"object_{obj}" if obj > 0 else "collision"
        elif event == "result_onset":
            label = {1: "success", 2: "failure"}.get(int(row["code"]), "incomplete")
        else:
            label = event
        semantic_rows.append({
            "trial_id": int(row["trial"]),
            "event": event,
            "label": label,
            "task_sample": task_sample,
            "task_time_s": float(row["task_time_s"]),
            "source_clock_ms": float(row["source_clock_ms"]),
            "eeg_sample": int(row["eeg_sample"]),
            "eeg_time_s": float(row["eeg_time_s"]),
            "alignment_error_ms": float(row["alignment_error_ms"]),
        })
    return pd.DataFrame(semantic_rows, columns=_EVENT_COLUMNS)


def _recording_id(path: Path) -> str:
    match = re.search(r"R\d+", path.name, re.IGNORECASE)
    return match.group(0).upper() if match else path.stem


def build_flight_trial_interface(
    eeg: BCI2000Dat,
    task: BCI2000Dat,
    *,
    segment: str = "flight",
    channel_indices: Iterable[int] | None = None,
    drop_empty_channels: bool = True,
) -> dict[str, Any]:
    """Build an in-memory ``flight_split`` from opened BCI2000 recordings.

    ``segment='flight'`` starts at ``flight_start`` and ends just before the
    result display; the collision sample is retained for failed trials.  This
    is the closest analogue of playgroundgit's true pen-down segment.
    """
    _require_states(eeg, ("SourceTime",))
    _require_states(task, ("SourceTime", "GamePhase", "Collision", "FlightTrialResult"))
    selected_channels, channel_names = _resolve_channels(
        eeg, channel_indices, drop_empty_channels
    )
    eeg_clock, task_clock, force_at_eeg_rate = align_force_to_eeg(eeg, task)
    events = _semantic_events(eeg, task, eeg_clock, task_clock)
    bounds = _task_trial_bounds(task, segment)
    recording_id = _recording_id(Path(eeg.path))

    continuous_state_values = {
        name: task.state(name).astype(np.float64) * scale + offset
        for name, (scale, offset) in _CONTINUOUS_STATES.items()
        if name in task.state_definitions
    }
    discrete_state_values = {
        name: task.state(name).astype(np.int64)
        for name in _DISCRETE_STATES
        if name in task.state_definitions
    }

    X_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    force_list: list[np.ndarray] = []
    force_normalized_list: list[np.ndarray] = []
    time_list: list[np.ndarray] = []
    source_time_ms_list: list[np.ndarray] = []
    state_list: list[dict[str, np.ndarray]] = []
    event_list: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []

    for bound in bounds:
        mapped = map_task_samples_to_eeg(
            np.asarray([bound["task_start"], bound["task_stop"] - 1], dtype=int),
            eeg_clock,
            task_clock,
        )
        eeg_start = int(mapped[0])
        eeg_stop = min(eeg.samples, int(mapped[1]) + 1)
        if eeg_stop <= eeg_start:
            continue
        query_clock = eeg_clock[eeg_start:eeg_stop]
        eeg_segment = np.asarray(
            eeg.signals[eeg_start:eeg_stop][:, selected_channels], dtype=np.float32
        ).T
        force_segment = np.asarray(force_at_eeg_rate[eeg_start:eeg_stop], dtype=np.float32)

        aligned_states: dict[str, np.ndarray] = {}
        for name, values in continuous_state_values.items():
            aligned_states[name] = np.interp(query_clock, task_clock, values).astype(np.float32)
        for name, values in discrete_state_values.items():
            aligned_states[name] = _nearest_task_values(values, query_clock, task_clock).astype(np.int64)

        normalized = aligned_states.get("GripForceNormalized")
        if normalized is None:
            normalized = np.full(force_segment.shape, np.nan, dtype=np.float32)
        relative_time = np.arange(eeg_segment.shape[1], dtype=np.float64) / eeg.sampling_rate
        relative_clock = query_clock - query_clock[0]
        trial_events = events.loc[events["trial_id"] == bound["trial_id"]].reset_index(drop=True)

        X_list.append(eeg_segment)
        target_list.append(force_segment[np.newaxis, :])
        force_list.append(force_segment)
        force_normalized_list.append(normalized)
        time_list.append(relative_time)
        source_time_ms_list.append(relative_clock)
        state_list.append(aligned_states)
        event_list.append(trial_events)
        meta_rows.append({
            "recording_id": recording_id,
            "trial_key": f"{recording_id}_trial-{bound['trial_id']:03d}",
            "trial_index0": bound["trial_index0"],
            "trial_id": bound["trial_id"],
            "segment": segment,
            "outcome": bound["outcome"],
            "collision": bound["collision"],
            "task_start_sample": bound["task_start"],
            "task_stop_sample": bound["task_stop"],
            "eeg_start_sample": eeg_start,
            "eeg_stop_sample": eeg_stop,
            "flight_start_task_sample": bound["flight_start"],
            "collision_task_sample": bound["collision_sample"],
            "result_task_sample": bound["result_sample"],
            "n_samples": eeg_segment.shape[1],
            "duration_s": eeg_segment.shape[1] / eeg.sampling_rate,
            "force_mean": float(np.nanmean(force_segment)),
            "force_std": float(np.nanstd(force_segment)),
            "force_min": float(np.nanmin(force_segment)),
            "force_max": float(np.nanmax(force_segment)),
        })

    meta = pd.DataFrame(meta_rows)
    return {
        "interface_version": INTERFACE_VERSION,
        "split_name": "all",
        "segment": segment,
        "X_list": X_list,
        "target_list": target_list,
        "target_names": ("grip_force_raw",),
        "target_units": ("source_signal_units",),
        "force_list": force_list,
        "force_normalized_list": force_normalized_list,
        "meta": meta,
        "events": events,
        "event_list": event_list,
        "sr": float(eeg.sampling_rate),
        "time_list": time_list,
        "source_time_ms_list": source_time_ms_list,
        "state_list": state_list,
        "state_names": tuple([*continuous_state_values, *discrete_state_values]),
        "game_phase_labels": dict(GAME_PHASE_LABELS),
        "channel_indices": selected_channels,
        "channel_names": channel_names,
        "raw_channel_names": list(eeg.channel_names),
        "n_ch": int(len(selected_channels)),
        "raw_n_ch": int(eeg.source_channels),
        "recording_id": recording_id,
        "eeg_path": Path(eeg.path),
        "task_path": Path(task.path),
    }


def select_flight_split(flight_split: dict[str, Any], split_name: str) -> dict[str, Any]:
    """Return one outcome split while preserving original trial identifiers."""
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
    meta = flight_split["meta"]
    if split_name == "all":
        mask = np.ones(len(meta), dtype=bool)
    elif split_name == "collision":
        mask = meta["collision"].to_numpy(dtype=bool)
    else:
        mask = (meta["outcome"] == split_name).to_numpy()
    selected_rows = np.flatnonzero(mask)
    selected = dict(flight_split)
    for key in _TRIAL_LIST_KEYS:
        selected[key] = [flight_split[key][i] for i in selected_rows]
    selected["meta"] = meta.loc[mask].reset_index(drop=True)
    selected_trial_ids = set(selected["meta"]["trial_id"].astype(int))
    selected["events"] = flight_split["events"].loc[
        flight_split["events"]["trial_id"].isin(selected_trial_ids)
    ].reset_index(drop=True)
    selected["split_name"] = split_name
    return selected


def load_all_flight_trials(
    eeg_path: str | Path,
    task_path: str | Path,
    *,
    segment: str = "flight",
    channel_indices: Iterable[int] | None = None,
    drop_empty_channels: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load the files once and return ``all/success/failure/collision`` splits."""
    eeg = BCI2000Dat(eeg_path)
    task = BCI2000Dat(task_path)
    all_trials = build_flight_trial_interface(
        eeg,
        task,
        segment=segment,
        channel_indices=channel_indices,
        drop_empty_channels=drop_empty_channels,
    )
    return {name: select_flight_split(all_trials, name) for name in SPLIT_NAMES}


def load_flight_trials(
    eeg_path: str | Path,
    task_path: str | Path,
    *,
    split_name: str = "all",
    segment: str = "flight",
    channel_indices: Iterable[int] | None = None,
    drop_empty_channels: bool = True,
) -> dict[str, Any]:
    """Load one team-facing trial split.

    This mirrors playgroundgit's ``load_pen_trials`` entry point.  For batch
    work, prefer :func:`load_all_flight_trials` so the recordings are read once.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
    splits = load_all_flight_trials(
        eeg_path,
        task_path,
        segment=segment,
        channel_indices=channel_indices,
        drop_empty_channels=drop_empty_channels,
    )
    return splits[split_name]


__all__ = [
    "INTERFACE_VERSION",
    "GAME_PHASE_LABELS",
    "SEGMENT_NAMES",
    "SPLIT_NAMES",
    "build_flight_trial_interface",
    "load_all_flight_trials",
    "load_flight_trials",
    "select_flight_split",
]
