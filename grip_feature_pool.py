"""Literature-informed EEG feature pool for aligned grip-force trials.

This module consumes one or more ``flight_split`` dictionaries returned by
``grip_data_interface.py``.  It performs configurable re-referencing, notch
filtering, causal/offline band filtering, historical windowing, time-domain,
band-power, spectral-shape and oscillatory-burst extraction, and label
generation.  The resulting :class:`FeaturePool` is independent of the
original BCI2000 files and can be handed directly to downstream modelling
code.

The named recipes reproduce the *frequency-band definitions* summarized in
the project literature review.  They are not claims of exact paper
reproduction: windowing, referencing, filtering and validation are recorded
separately in ``manifest`` and remain explicit arguments here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos


FEATURE_INTERFACE_VERSION = "1.1"

TIME_DOMAIN_FEATURES = (
    "lmp",
    "slope",
    "rms",
    "line_length",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
)
SPECTRAL_SHAPE_FEATURES = ("spectral_entropy", "spectral_centroid")
SPECTRAL_SHAPE_BANDS_HZ = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 60.0),
    (60.0, 150.0),
    (150.0, 300.0),
)
BURST_METRICS = ("occupancy", "rate", "mean_duration")

# Frequency bands are kept study-specific on purpose.  Overlapping bands are
# useful for direct feature-family ablations and regularized models.
FEATURE_RECIPES: dict[str, dict[str, Any]] = {
    "flint2014": {
        "lmp": True,
        "bands_hz": ((0.0, 4.0), (7.0, 20.0), (70.0, 115.0),
                     (130.0, 200.0), (200.0, 300.0)),
    },
    "flint2020": {
        "lmp": False,
        "bands_hz": ((8.0, 55.0), (70.0, 150.0)),
    },
    "wu2022": {
        "lmp": False,
        "bands_hz": ((0.5, 4.0), (4.0, 13.0), (13.0, 30.0),
                     (30.0, 60.0), (60.0, 150.0)),
    },
    "merk2022": {
        "lmp": False,
        "bands_hz": ((4.0, 8.0), (8.0, 12.0), (13.0, 35.0),
                     (13.0, 20.0), (20.0, 35.0), (60.0, 80.0),
                     (90.0, 200.0), (60.0, 200.0)),
    },
    "pistohl2012": {
        "lmp": True,  # The paper calls the low-frequency voltage feature LFC.
        "bands_hz": ((0.0, 10.0), (14.0, 26.0), (74.0, 118.0)),
    },
    "jiang2020": {
        "lmp": False,
        "bands_hz": ((8.0, 32.0), (60.0, 200.0)),
    },
    "compact": {
        "lmp": True,
        "bands_hz": ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0),
                     (13.0, 30.0), (30.0, 60.0), (60.0, 150.0),
                     (150.0, 300.0)),
    },
}


def _literature_union() -> tuple[tuple[float, float], ...]:
    seen: set[tuple[float, float]] = set()
    bands: list[tuple[float, float]] = []
    for name in (
        "flint2014", "flint2020", "wu2022", "merk2022",
        "pistohl2012", "jiang2020",
    ):
        for band in FEATURE_RECIPES[name]["bands_hz"]:
            normalized = (float(band[0]), float(band[1]))
            if normalized not in seen:
                seen.add(normalized)
                bands.append(normalized)
    return tuple(bands)


FEATURE_RECIPES["literature_all"] = {
    "lmp": True,
    "bands_hz": _literature_union(),
}

# A practical union for the present dataset.  All features end at the same
# feature time, but slower features use longer history than fast activity.
# This keeps X as window x channel x feature while respecting distinct time
# scales.  Connectivity/PAC are intentionally separate future interfaces
# because their natural axis is channel-pair rather than channel.
FEATURE_RECIPES["expanded_multiscale"] = {
    "lmp": False,
    "bands_hz": _literature_union(),
    "time_features": TIME_DOMAIN_FEATURES,
    "spectral_features": SPECTRAL_SHAPE_FEATURES,
    "spectral_shape_bands_hz": SPECTRAL_SHAPE_BANDS_HZ,
    "burst_bands": (
        {"name": "beta", "low_hz": 13.0, "high_hz": 30.0, "envelope_ms": 100.0},
        {"name": "high_gamma", "low_hz": 70.0, "high_hz": 150.0, "envelope_ms": 25.0},
    ),
    "window_ms": {
        "lmp": 2000.0,
        "time_domain": 500.0,
        "low_frequency": 2000.0,
        "mid_frequency": 500.0,
        "high_frequency": 250.0,
        "spectral_shape": 1000.0,
        "burst": 500.0,
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _parquet_available() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


@dataclass
class FeaturePool:
    """Materialized feature array plus labels, window index and provenance."""

    X: np.ndarray
    labels: pd.DataFrame
    windows: pd.DataFrame
    feature_info: dict[str, Any]
    manifest: dict[str, Any]
    raw_windows: np.ndarray | None = None

    def validate(self) -> None:
        """Raise if array axes and metadata tables are inconsistent."""
        if self.X.ndim != 3:
            raise ValueError("X must have shape (n_windows, n_channels, n_features)")
        n_windows, n_channels, n_features = self.X.shape
        if len(self.labels) != n_windows or len(self.windows) != n_windows:
            raise ValueError("X, labels and windows must contain the same number of rows")
        if len(self.feature_info.get("channel_axis", [])) != n_channels:
            raise ValueError("channel_axis length does not match X.shape[1]")
        if len(self.feature_info.get("feature_axis", [])) != n_features:
            raise ValueError("feature_axis length does not match X.shape[2]")
        expected_ids = np.arange(n_windows)
        for table_name, table in (("labels", self.labels), ("windows", self.windows)):
            if "window_id" not in table:
                raise ValueError(f"{table_name} is missing window_id")
            if not np.array_equal(table["window_id"].to_numpy(), expected_ids):
                raise ValueError(f"{table_name}.window_id is not aligned with X")
        if self.raw_windows is not None:
            if self.raw_windows.ndim != 3:
                raise ValueError("raw_windows must be 3-D")
            if self.raw_windows.shape[:2] != self.X.shape[:2]:
                raise ValueError("raw_windows window/channel axes must match X")

    @property
    def feature_names(self) -> list[str]:
        return [item["name"] for item in self.feature_info["feature_axis"]]

    @property
    def channel_names(self) -> list[str]:
        return [item["name"] for item in self.feature_info["channel_axis"]]

    def as_sklearn(
        self,
        *,
        target: str = "force_normalized",
        mask: str | Sequence[bool] | np.ndarray | None = "mask_flight",
        features: Sequence[str] | None = None,
        group_column: str = "trial_key",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return flattened ``X``, one target ``y`` and leakage-safe groups.

        ``mask`` may name a boolean column in ``windows``/``labels`` or provide
        a boolean vector.  Grouping defaults to complete trials, which should
        be passed to GroupKFold or another group-aware splitter.
        """
        self.validate()
        if target not in self.labels:
            raise KeyError(f"Unknown target {target!r}; choose from {list(self.labels)}")
        if group_column not in self.windows:
            raise KeyError(f"Unknown group column {group_column!r}")

        if mask is None:
            row_mask = np.ones(len(self.windows), dtype=bool)
        elif isinstance(mask, str):
            source = self.windows if mask in self.windows else self.labels
            if mask not in source:
                raise KeyError(f"Unknown mask column {mask!r}")
            row_mask = source[mask].to_numpy(dtype=bool)
        else:
            row_mask = np.asarray(mask, dtype=bool)
            if row_mask.shape != (len(self.windows),):
                raise ValueError("mask must have one boolean value per window")

        if features is None:
            feature_indices = np.arange(self.X.shape[2])
        else:
            by_name = {name: i for i, name in enumerate(self.feature_names)}
            missing = [name for name in features if name not in by_name]
            if missing:
                raise KeyError(f"Unknown feature names: {missing}")
            feature_indices = np.asarray([by_name[name] for name in features], dtype=int)

        selected = self.X[row_mask][:, :, feature_indices]
        X_flat = selected.reshape(selected.shape[0], -1)
        y = self.labels.loc[row_mask, target].to_numpy()
        groups = self.windows.loc[row_mask, group_column].to_numpy()
        return X_flat, y, groups

    def save(self, output_dir: str | Path, *, overwrite: bool = False) -> Path:
        """Write the five-file feature dataset contract to ``output_dir``.

        Parquet output requires either ``pyarrow`` or ``fastparquet``.  The
        dependency is checked before any file is written, so a missing engine
        cannot leave a partially generated artifact.
        """
        self.validate()
        if not _parquet_available():
            raise ModuleNotFoundError(
                "Saving labels.parquet/windows.parquet requires pyarrow or "
                "fastparquet. Install one in the active environment, e.g. "
                "`conda install -n eeg pyarrow`."
            )
        output_dir = Path(output_dir)
        expected = (
            output_dir / "manifest.json",
            output_dir / "features.npz",
            output_dir / "labels.parquet",
            output_dir / "windows.parquet",
            output_dir / "feature_names.json",
        )
        existing = [path for path in expected if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"Feature dataset files already exist in {output_dir}; "
                "pass overwrite=True to replace them"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        arrays = {"X": np.asarray(self.X)}
        if self.raw_windows is not None:
            arrays["raw_windows"] = np.asarray(self.raw_windows)
        np.savez_compressed(output_dir / "features.npz", **arrays)
        self.labels.to_parquet(output_dir / "labels.parquet", index=False)
        self.windows.to_parquet(output_dir / "windows.parquet", index=False)
        (output_dir / "feature_names.json").write_text(
            json.dumps(self.feature_info, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return output_dir


def _as_split_list(
    flight_splits: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(flight_splits, Mapping) and "X_list" in flight_splits:
        splits = [flight_splits]
    elif isinstance(flight_splits, Mapping):
        splits = list(flight_splits.values())
    else:
        splits = list(flight_splits)
    if not splits:
        raise ValueError("At least one flight_split is required")
    for split in splits:
        if "X_list" not in split:
            raise TypeError("Each input must be a flight_split containing X_list")
    return splits


def _validate_splits(splits: Sequence[Mapping[str, Any]]) -> tuple[float, list[str]]:
    sr = float(splits[0]["sr"])
    channel_names = list(splits[0]["channel_names"])
    for split in splits:
        if not np.isclose(float(split["sr"]), sr):
            raise ValueError("All recordings must have the same EEG sampling rate")
        if list(split["channel_names"]) != channel_names:
            raise ValueError(
                "All recordings must use identical channel names and order; "
                "select/reorder channels before building a combined pool"
            )
        n_trials = len(split["X_list"])
        for key in (
            "target_list", "force_normalized_list", "time_list", "state_list",
        ):
            if len(split[key]) != n_trials:
                raise ValueError(f"{key} is not aligned with X_list")
        if len(split["meta"]) != n_trials:
            raise ValueError("meta is not aligned with X_list")
    return sr, channel_names


def _apply_sos(signal: np.ndarray, sos: np.ndarray, causal: bool) -> np.ndarray:
    if causal:
        zi_base = sosfilt_zi(sos)
        zi = zi_base[:, np.newaxis, :] * signal[np.newaxis, :, :1]
        filtered, _ = sosfilt(sos, signal, axis=-1, zi=zi)
        return filtered
    return sosfiltfilt(sos, signal, axis=-1)


def _preprocess_signal(
    signal: np.ndarray,
    sr: float,
    *,
    reference: str,
    notch_hz: Sequence[float],
    notch_quality: float,
    causal: bool,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Each EEG trial must have shape (n_channels, n_samples)")
    if not np.all(np.isfinite(x)):
        raise ValueError("EEG contains NaN or infinite samples")
    if reference == "common_average":
        x = x - x.mean(axis=0, keepdims=True)
    elif reference == "median":
        x = x - np.median(x, axis=0, keepdims=True)
    elif reference != "none":
        raise ValueError("reference must be 'none', 'common_average', or 'median'")

    nyquist = sr / 2.0
    for frequency in notch_hz:
        frequency = float(frequency)
        if frequency <= 0 or frequency >= nyquist:
            continue
        b, a = iirnotch(frequency, notch_quality, fs=sr)
        x = _apply_sos(x, tf2sos(b, a), causal)
    return x


def _band_sos(low_hz: float, high_hz: float, sr: float, order: int) -> np.ndarray:
    nyquist = sr / 2.0
    if low_hz < 0 or high_hz <= low_hz:
        raise ValueError(f"Invalid frequency band ({low_hz}, {high_hz})")
    if low_hz == 0:
        if high_hz >= nyquist:
            raise ValueError("A 0-to-Nyquist band is not a meaningful band-power feature")
        return butter(order, high_hz, btype="lowpass", fs=sr, output="sos")
    if high_hz >= nyquist:
        raise ValueError(
            f"Band ({low_hz:g}, {high_hz:g}) Hz reaches/exceeds Nyquist "
            f"({nyquist:g} Hz)"
        )
    return butter(order, (low_hz, high_hz), btype="bandpass", fs=sr, output="sos")


def _window_means(values: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(
        [np.zeros((values.shape[0], 1), dtype=np.float64),
         np.cumsum(values, axis=-1, dtype=np.float64)],
        axis=-1,
    )
    totals = cumulative[:, stops] - cumulative[:, starts]
    return (totals / (stops - starts)[np.newaxis, :]).T


def _window_sums(values: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(
        [np.zeros((values.shape[0], 1), dtype=np.float64),
         np.cumsum(values, axis=-1, dtype=np.float64)],
        axis=-1,
    )
    return (cumulative[:, stops] - cumulative[:, starts]).T


def _feature_starts(stops: np.ndarray, window_ms: float, sr: float) -> np.ndarray:
    samples = int(round(float(window_ms) * sr / 1000.0))
    if samples < 2:
        raise ValueError(f"Feature window {window_ms:g} ms is too short at {sr:g} Hz")
    starts = stops - samples
    if np.any(starts < 0):
        raise ValueError("Output windows do not contain enough history for every feature")
    return starts


def _window_variance(values: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    mean = _window_means(values, starts, stops)
    mean_square = _window_means(values * values, starts, stops)
    return np.maximum(mean_square - mean * mean, 0.0)


def _window_slopes(
    values: np.ndarray, starts: np.ndarray, stops: np.ndarray, sr: float
) -> np.ndarray:
    """Least-squares linear slope in source-signal units per second."""
    sample_index = np.arange(values.shape[1], dtype=np.float64)
    cumulative_x = np.concatenate(
        [np.zeros((values.shape[0], 1)), np.cumsum(values, axis=-1)], axis=-1
    )
    cumulative_ix = np.concatenate(
        [np.zeros((values.shape[0], 1)),
         np.cumsum(values * sample_index[np.newaxis, :], axis=-1)],
        axis=-1,
    )
    sum_x = (cumulative_x[:, stops] - cumulative_x[:, starts]).T
    sum_ix = (cumulative_ix[:, stops] - cumulative_ix[:, starts]).T
    n = (stops - starts).astype(np.float64)
    sum_i = (starts + stops - 1) * n / 2.0
    sum_i2 = (
        (stops - 1) * stops * (2 * stops - 1)
        - (starts - 1) * starts * (2 * starts - 1)
    ) / 6.0
    denominator = n * sum_i2 - sum_i * sum_i
    slope_per_sample = (n[:, None] * sum_ix - sum_i[:, None] * sum_x) / denominator[:, None]
    return slope_per_sample * sr


def _window_hjorth(
    values: np.ndarray, starts: np.ndarray, stops: np.ndarray, sr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activity = _window_variance(values, starts, stops)
    first = np.diff(values, axis=-1) * sr
    second = np.diff(first, axis=-1) * sr
    first_var = _window_variance(first, starts, stops - 1)
    second_var = _window_variance(second, starts, stops - 2)
    eps = np.finfo(np.float64).eps
    mobility = np.sqrt(first_var / np.maximum(activity, eps))
    derivative_mobility = np.sqrt(second_var / np.maximum(first_var, eps))
    complexity = derivative_mobility / np.maximum(mobility, eps)
    return activity, mobility, complexity


def _causal_moving_mean(values: np.ndarray, samples: int) -> np.ndarray:
    """Trailing moving mean with a progressively growing initial window."""
    samples = int(samples)
    if samples < 1:
        raise ValueError("Moving-average length must be positive")
    cumulative = np.concatenate(
        [np.zeros((values.shape[0], 1), dtype=np.float64),
         np.cumsum(values, axis=-1, dtype=np.float64)],
        axis=-1,
    )
    stops = np.arange(1, values.shape[1] + 1)
    starts = np.maximum(0, stops - samples)
    return (
        cumulative[:, stops] - cumulative[:, starts]
    ) / (stops - starts)[None, :]


def _burst_window_metrics(
    band_power_samples: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    *,
    sr: float,
    baseline_stop: int,
    threshold_mad: float,
    envelope_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return burst occupancy, rate and approximate mean duration.

    The fixed per-channel threshold is median + ``threshold_mad`` robust SD,
    estimated only from the history preceding the first output label.  This
    prevents thresholds from looking ahead into later parts of a trial.
    """
    envelope_samples = max(1, int(round(envelope_ms * sr / 1000.0)))
    envelope = _causal_moving_mean(band_power_samples, envelope_samples)
    baseline = envelope[:, :baseline_stop]
    median = np.median(baseline, axis=-1)
    mad = np.median(np.abs(baseline - median[:, None]), axis=-1)
    robust_sd = 1.4826 * mad
    fallback = np.std(baseline, axis=-1)
    scale = np.where(robust_sd > 0, robust_sd, fallback)
    threshold = median + threshold_mad * np.maximum(scale, np.finfo(float).eps)
    active = envelope > threshold[:, None]
    occupancy = _window_means(active.astype(np.float64), starts, stops)

    onset = active & np.concatenate(
        [np.ones((active.shape[0], 1), dtype=bool), ~active[:, :-1]], axis=-1
    )
    onset_count = _window_sums(onset.astype(np.float64), starts, stops)
    # A burst that started before the window still counts as one active burst.
    carried = np.stack(
        [active[:, start] & ~onset[:, start] for start in starts], axis=0
    ).astype(float)
    burst_count = onset_count + carried
    duration_s = (stops - starts) / sr
    rate = burst_count / duration_s[:, None]
    mean_duration = occupancy * duration_s[:, None] / np.maximum(burst_count, 1.0)
    mean_duration[occupancy == 0] = 0.0
    return occupancy, rate, mean_duration


def _format_frequency(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p")


def _band_feature_name(low_hz: float, high_hz: float) -> str:
    return f"bandpower_{_format_frequency(low_hz)}_{_format_frequency(high_hz)}Hz"


def _recipe_sources(low_hz: float, high_hz: float) -> list[str]:
    band = (float(low_hz), float(high_hz))
    return [
        name for name, recipe in FEATURE_RECIPES.items()
        if name != "literature_all" and band in recipe["bands_hz"]
    ]


def _effective_windows_ms(recipe_name: str, default_window_ms: float) -> dict[str, float]:
    configured = FEATURE_RECIPES[recipe_name].get("window_ms", {})
    return {
        "lmp": float(configured.get("lmp", default_window_ms)),
        "time_domain": float(configured.get("time_domain", default_window_ms)),
        "low_frequency": float(configured.get("low_frequency", default_window_ms)),
        "mid_frequency": float(configured.get("mid_frequency", default_window_ms)),
        "high_frequency": float(configured.get("high_frequency", default_window_ms)),
        "spectral_shape": float(configured.get("spectral_shape", default_window_ms)),
        "burst": float(configured.get("burst", default_window_ms)),
    }


def _band_window_ms(low_hz: float, windows_ms: Mapping[str, float]) -> float:
    if low_hz < 4.0:
        return windows_ms["low_frequency"]
    if low_hz >= 60.0:
        return windows_ms["high_frequency"]
    return windows_ms["mid_frequency"]


def _feature_axis(
    recipe_name: str, log_power: bool, default_window_ms: float
) -> list[dict[str, Any]]:
    recipe = FEATURE_RECIPES[recipe_name]
    windows_ms = _effective_windows_ms(recipe_name, default_window_ms)
    axis: list[dict[str, Any]] = []
    time_features = tuple(recipe.get("time_features", ()))
    if recipe["lmp"] and "lmp" not in time_features:
        sources = [
            name for name, item in FEATURE_RECIPES.items()
            if name != "literature_all" and item["lmp"]
        ]
        axis.append({
            "index": len(axis),
            "name": "lmp",
            "family": "time_domain",
            "unit": "source_signal_units",
            "recipe_sources": sources,
            "window_ms": windows_ms["lmp"],
        })
    time_units = {
        "lmp": "source_signal_units",
        "slope": "source_signal_units/s",
        "rms": "source_signal_units",
        "line_length": "source_signal_units/s",
        "hjorth_activity": "source_signal_units^2",
        "hjorth_mobility": "Hz_like",
        "hjorth_complexity": "dimensionless",
    }
    for name in time_features:
        axis.append({
            "index": len(axis),
            "name": name,
            "family": "time_domain",
            "unit": time_units[name],
            "recipe_sources": [],
            "window_ms": windows_ms["lmp"] if name == "lmp" else windows_ms["time_domain"],
        })
    for low_hz, high_hz in recipe["bands_hz"]:
        axis.append({
            "index": len(axis),
            "name": _band_feature_name(low_hz, high_hz),
            "family": "band_power",
            "low_hz": float(low_hz),
            "high_hz": float(high_hz),
            "unit": "dB(source_signal_units^2)" if log_power else "source_signal_units^2",
            "recipe_sources": _recipe_sources(low_hz, high_hz),
            "window_ms": _band_window_ms(float(low_hz), windows_ms),
        })
    for name in recipe.get("spectral_features", ()):
        axis.append({
            "index": len(axis),
            "name": name,
            "family": "spectral_shape",
            "unit": "dimensionless" if name == "spectral_entropy" else "Hz",
            "recipe_sources": [],
            "window_ms": windows_ms["spectral_shape"],
            "bands_hz": [list(band) for band in recipe["spectral_shape_bands_hz"]],
        })
    burst_units = {
        "occupancy": "fraction",
        "rate": "bursts/s",
        "mean_duration": "s",
    }
    for band in recipe.get("burst_bands", ()):
        for metric in BURST_METRICS:
            axis.append({
                "index": len(axis),
                "name": f"burst_{band['name']}_{metric}",
                "family": "burst",
                "unit": burst_units[metric],
                "low_hz": float(band["low_hz"]),
                "high_hz": float(band["high_hz"]),
                "envelope_ms": float(band["envelope_ms"]),
                "recipe_sources": [],
                "window_ms": windows_ms["burst"],
            })
    return axis


def _extract_trial_features(
    signal: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    *,
    sr: float,
    recipe_name: str,
    reference: str,
    notch_hz: Sequence[float],
    notch_quality: float,
    causal: bool,
    filter_order: int,
    log_power: bool,
    power_floor: float,
    default_window_ms: float,
    burst_threshold_mad: float,
) -> tuple[np.ndarray, np.ndarray]:
    processed = _preprocess_signal(
        signal,
        sr,
        reference=reference,
        notch_hz=notch_hz,
        notch_quality=notch_quality,
        causal=causal,
    )
    feature_blocks: list[np.ndarray] = []
    recipe = FEATURE_RECIPES[recipe_name]
    windows_ms = _effective_windows_ms(recipe_name, default_window_ms)
    time_features = tuple(recipe.get("time_features", ()))
    if recipe["lmp"] and "lmp" not in time_features:
        local_starts = _feature_starts(stops, windows_ms["lmp"], sr)
        feature_blocks.append(_window_means(processed, local_starts, stops))

    if time_features:
        time_starts = _feature_starts(stops, windows_ms["time_domain"], sr)
        lmp_starts = _feature_starts(stops, windows_ms["lmp"], sr)
        time_cache: dict[str, np.ndarray] = {
            "lmp": _window_means(processed, lmp_starts, stops),
            "slope": _window_slopes(processed, time_starts, stops, sr),
            "rms": np.sqrt(_window_means(processed * processed, time_starts, stops)),
            "line_length": _window_means(
                np.abs(np.diff(processed, axis=-1)) * sr, time_starts, stops - 1
            ),
        }
        activity, mobility, complexity = _window_hjorth(
            processed, time_starts, stops, sr
        )
        time_cache.update({
            "hjorth_activity": activity,
            "hjorth_mobility": mobility,
            "hjorth_complexity": complexity,
        })
        feature_blocks.extend(time_cache[name] for name in time_features)

    filtered_power_samples: dict[tuple[float, float], np.ndarray] = {}
    for low_hz, high_hz in recipe["bands_hz"]:
        filtered = _apply_sos(
            processed,
            _band_sos(float(low_hz), float(high_hz), sr, filter_order),
            causal,
        )
        power_samples = filtered * filtered
        filtered_power_samples[(float(low_hz), float(high_hz))] = power_samples
        local_starts = _feature_starts(
            stops, _band_window_ms(float(low_hz), windows_ms), sr
        )
        power = _window_means(power_samples, local_starts, stops)
        if log_power:
            power = 10.0 * np.log10(np.maximum(power, power_floor))
        feature_blocks.append(power)

    spectral_features = tuple(recipe.get("spectral_features", ()))
    if spectral_features:
        spectral_starts = _feature_starts(stops, windows_ms["spectral_shape"], sr)
        spectral_powers: list[np.ndarray] = []
        centers: list[float] = []
        for low_hz, high_hz in recipe["spectral_shape_bands_hz"]:
            band = (float(low_hz), float(high_hz))
            power_samples = filtered_power_samples.get(band)
            if power_samples is None:
                filtered = _apply_sos(
                    processed,
                    _band_sos(*band, sr, filter_order),
                    causal,
                )
                power_samples = filtered * filtered
                filtered_power_samples[band] = power_samples
            spectral_powers.append(
                _window_means(power_samples, spectral_starts, stops)
            )
            centers.append((band[0] + band[1]) / 2.0)
        power_stack = np.stack(spectral_powers, axis=-1)
        total_power = np.maximum(power_stack.sum(axis=-1), power_floor)
        proportions = power_stack / total_power[..., None]
        entropy = -np.sum(
            proportions * np.log(np.maximum(proportions, power_floor)), axis=-1
        ) / np.log(len(spectral_powers))
        centroid = np.sum(
            proportions * np.asarray(centers)[None, None, :], axis=-1
        )
        spectral_cache = {
            "spectral_entropy": entropy,
            "spectral_centroid": centroid,
        }
        feature_blocks.extend(spectral_cache[name] for name in spectral_features)

    burst_bands = tuple(recipe.get("burst_bands", ()))
    if burst_bands:
        burst_starts = _feature_starts(stops, windows_ms["burst"], sr)
        baseline_stop = int(stops[0])
        for band_config in burst_bands:
            band = (float(band_config["low_hz"]), float(band_config["high_hz"]))
            power_samples = filtered_power_samples.get(band)
            if power_samples is None:
                filtered = _apply_sos(
                    processed,
                    _band_sos(*band, sr, filter_order),
                    causal,
                )
                power_samples = filtered * filtered
                filtered_power_samples[band] = power_samples
            feature_blocks.extend(_burst_window_metrics(
                power_samples,
                burst_starts,
                stops,
                sr=sr,
                baseline_stop=baseline_stop,
                threshold_mad=burst_threshold_mad,
                envelope_ms=float(band_config["envelope_ms"]),
            ))
    # Each block is windows x channels. Stack features last.
    return np.stack(feature_blocks, axis=-1).astype(np.float32), processed


def _force_activity(
    normalized: np.ndarray,
    raw: np.ndarray,
    threshold: float,
) -> np.ndarray:
    finite_normalized = np.isfinite(normalized)
    if finite_normalized.any():
        activity = np.zeros(len(raw), dtype=bool)
        activity[finite_normalized] = normalized[finite_normalized] > threshold
        if finite_normalized.all():
            return activity
    else:
        activity = np.zeros(len(raw), dtype=bool)

    low, high = np.nanpercentile(raw, (1.0, 99.0))
    scale = high - low
    fallback = np.zeros(len(raw), dtype=bool) if scale <= 0 else (raw - low) / scale > threshold
    activity[~finite_normalized] = fallback[~finite_normalized]
    return activity


def build_grip_feature_pool(
    flight_splits: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    recipe: str = "literature_all",
    window_ms: float = 500.0,
    step_ms: float = 50.0,
    label_lag_ms: float = 0.0,
    causal: bool = True,
    reference: str = "none",
    notch_hz: Sequence[float] = (50.0, 100.0, 150.0, 200.0, 250.0),
    notch_quality: float = 30.0,
    filter_order: int = 4,
    log_power: bool = True,
    power_floor: float = 1e-20,
    force_active_threshold: float = 0.05,
    burst_threshold_mad: float = 2.0,
    include_raw_windows: bool = False,
) -> FeaturePool:
    """Build a model-ready feature pool from one or more aligned recordings.

    Windows are historical: a window ``[start, stop)`` produces its label at
    ``stop - 1 + label_lag``.  With ``causal=True`` and non-negative label lag,
    no EEG sample after the feature time is used.  Use ``segment='trial'`` in
    the upstream interface when pre-flight history or Countdown baseline is
    required, and select ``mask_flight`` only after feature extraction.
    """
    if recipe not in FEATURE_RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; choose from {tuple(FEATURE_RECIPES)}")
    if window_ms <= 0 or step_ms <= 0:
        raise ValueError("window_ms and step_ms must be positive")
    if filter_order < 1:
        raise ValueError("filter_order must be at least 1")
    if notch_quality <= 0 or power_floor <= 0:
        raise ValueError("notch_quality and power_floor must be positive")
    if burst_threshold_mad <= 0:
        raise ValueError("burst_threshold_mad must be positive")
    if not 0 <= force_active_threshold <= 1:
        raise ValueError("force_active_threshold must be within 0..1")

    splits = _as_split_list(flight_splits)
    sr, channel_names = _validate_splits(splits)
    effective_windows_ms = _effective_windows_ms(recipe, window_ms)
    history_window_ms = max(effective_windows_ms.values())
    window_samples = int(round(history_window_ms * sr / 1000.0))
    step_samples = int(round(step_ms * sr / 1000.0))
    lag_samples = int(round(label_lag_ms * sr / 1000.0))
    if window_samples < 2 or step_samples < 1:
        raise ValueError("window_ms/step_ms are too short at this sampling rate")

    feature_axis = _feature_axis(recipe, log_power, window_ms)
    all_features: list[np.ndarray] = []
    all_raw_windows: list[np.ndarray] = []
    label_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    recording_ids: list[str] = []
    source_segments: list[str] = []
    window_id = 0

    for split in splits:
        recording_id = str(split.get("recording_id", "recording"))
        recording_ids.append(recording_id)
        source_segments.append(str(split.get("segment", "unknown")))
        phase_labels = {int(k): str(v) for k, v in split.get("game_phase_labels", {}).items()}

        for trial_i, signal in enumerate(split["X_list"]):
            signal = np.asarray(signal)
            n_samples = signal.shape[1]
            last_start = n_samples - window_samples
            if last_start < 0:
                continue
            starts = np.arange(0, last_start + 1, step_samples, dtype=int)
            stops = starts + window_samples
            label_indices = stops - 1 + lag_samples
            valid = (label_indices >= 0) & (label_indices < n_samples)
            starts, stops, label_indices = starts[valid], stops[valid], label_indices[valid]
            if starts.size == 0:
                continue

            trial_features, processed = _extract_trial_features(
                signal,
                starts,
                stops,
                sr=sr,
                recipe_name=recipe,
                reference=reference,
                notch_hz=notch_hz,
                notch_quality=notch_quality,
                causal=causal,
                filter_order=filter_order,
                log_power=log_power,
                power_floor=power_floor,
                default_window_ms=window_ms,
                burst_threshold_mad=burst_threshold_mad,
            )
            all_features.append(trial_features)
            if include_raw_windows:
                all_raw_windows.append(
                    np.stack([processed[:, start:stop] for start, stop in zip(starts, stops)])
                    .astype(np.float32)
                )

            raw_force = np.asarray(split["target_list"][trial_i], dtype=float).reshape(-1)
            normalized = np.asarray(split["force_normalized_list"][trial_i], dtype=float).reshape(-1)
            if len(raw_force) != n_samples or len(normalized) != n_samples:
                raise ValueError("Force labels are not sample-aligned with EEG")
            force_rate = np.gradient(raw_force, 1.0 / sr)
            force_acceleration = np.gradient(force_rate, 1.0 / sr)
            force_active = _force_activity(
                normalized, raw_force, force_active_threshold
            )

            states = split["state_list"][trial_i]
            phase = np.asarray(states.get("GamePhase", np.full(n_samples, -1)), dtype=int)
            collision = np.asarray(states.get("Collision", np.zeros(n_samples)), dtype=int)
            collision_onset = (collision > 0) & np.r_[True, collision[:-1] <= 0]
            # A point event will rarely land exactly on a downsampled label
            # index. Assign each onset to the first output window whose label
            # time follows it, covering one output step and producing one
            # positive label per isolated collision.
            collision_onset_at_window = np.asarray([
                collision_onset[max(0, int(index) - step_samples + 1):int(index) + 1].any()
                for index in label_indices
            ], dtype=bool)
            time_s = np.asarray(split["time_list"][trial_i], dtype=float)
            source_ms = np.asarray(
                split.get("source_time_ms_list", [time_s * 1000.0] * len(split["X_list"]))[trial_i],
                dtype=float,
            )
            meta = split["meta"].iloc[trial_i]
            trial_id = int(meta.get("trial_id", trial_i + 1))
            trial_key = str(meta.get("trial_key", f"{recording_id}_trial-{trial_id:03d}"))
            outcome = str(meta.get("outcome", "unknown"))
            trial_collision = bool(meta.get("collision", np.any(collision > 0)))
            global_eeg_start = int(meta.get("eeg_start_sample", 0))

            for start, stop, label_index, onset_in_step in zip(
                starts, stops, label_indices, collision_onset_at_window
            ):
                phase_value = int(phase[label_index])
                label_rows.append({
                    "window_id": window_id,
                    "force_raw": float(raw_force[label_index]),
                    "force_normalized": float(normalized[label_index]),
                    "force_derivative": float(force_rate[label_index]),
                    "force_acceleration": float(force_acceleration[label_index]),
                    "is_force_active": bool(force_active[label_index]),
                    "game_phase": phase_value,
                    "collision": bool(collision[label_index] > 0),
                    "collision_onset": bool(onset_in_step),
                    "outcome": outcome,
                    "trial_collision": trial_collision,
                })
                window_rows.append({
                    "window_id": window_id,
                    "recording_id": recording_id,
                    "trial_index0": int(meta.get("trial_index0", trial_i)),
                    "trial_id": trial_id,
                    "trial_key": trial_key,
                    "window_start_sample": int(start),
                    "window_stop_sample": int(stop),
                    "label_sample": int(label_index),
                    "global_eeg_label_sample": global_eeg_start + int(label_index),
                    "window_start_s": float(time_s[start]),
                    "window_end_s": float(time_s[start] + window_samples / sr),
                    "label_time_s": float(time_s[label_index]),
                    "source_clock_ms": float(source_ms[label_index]),
                    "outcome": outcome,
                    "trial_collision": trial_collision,
                    "game_phase_label": phase_labels.get(phase_value, f"unknown_{phase_value}"),
                    "mask_flight": bool(phase_value in (2, 3)),
                    "mask_playing": bool(phase_value == 2),
                    "mask_force_active": bool(force_active[label_index]),
                })
                window_id += 1

    if not all_features:
        raise ValueError("No complete windows could be extracted from the supplied trials")

    X = np.concatenate(all_features, axis=0)
    raw_windows = np.concatenate(all_raw_windows, axis=0) if include_raw_windows else None
    labels = pd.DataFrame(label_rows)
    windows = pd.DataFrame(window_rows)
    feature_info = {
        "feature_axis": feature_axis,
        "channel_axis": [
            {
                "index": i,
                "name": name,
                "original_index": int(splits[0].get("channel_indices", np.arange(len(channel_names)))[i]),
            }
            for i, name in enumerate(channel_names)
        ],
    }
    warnings: list[str] = []
    if any(segment != "trial" for segment in source_segments):
        warnings.append(
            "At least one upstream split was not segment='trial'; early windows "
            "may lack Countdown history/baseline."
        )
    if not causal:
        warnings.append("Offline zero-phase filtering uses future samples and is not online-safe.")
    if label_lag_ms < 0:
        warnings.append("Negative label_lag_ms predicts a label earlier than the feature-window end.")

    manifest = {
        "dataset_version": "1.0",
        "feature_interface_version": FEATURE_INTERFACE_VERSION,
        "source_interface_versions": sorted({
            str(split.get("interface_version", "unknown")) for split in splits
        }),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recording_ids": list(dict.fromkeys(recording_ids)),
        "source_segments": list(dict.fromkeys(source_segments)),
        "sampling_rate_hz": sr,
        "recipe": recipe,
        "window_ms": window_ms,
        "history_window_ms": history_window_ms,
        "feature_windows_ms": effective_windows_ms,
        "step_ms": step_ms,
        "label_lag_ms": label_lag_ms,
        "label_anchor": "window_stop_minus_1_plus_lag",
        "causal": causal,
        "reference": reference,
        "notch_hz": [float(value) for value in notch_hz if 0 < value < sr / 2.0],
        "notch_quality": notch_quality,
        "filter_order": filter_order,
        "log_power": log_power,
        "force_active_threshold": force_active_threshold,
        "burst_threshold_mad": burst_threshold_mad,
        "burst_threshold_baseline": "trial_history_before_first_output_label",
        "feature_shape": list(X.shape),
        "raw_windows_included": include_raw_windows,
        "target_names": [column for column in labels if column != "window_id"],
        "warnings": warnings,
    }
    pool = FeaturePool(
        X=X,
        labels=labels,
        windows=windows,
        feature_info=feature_info,
        manifest=manifest,
        raw_windows=raw_windows,
    )
    pool.validate()
    return pool


def load_feature_pool(input_dir: str | Path) -> FeaturePool:
    """Load a feature dataset written by :meth:`FeaturePool.save`."""
    if not _parquet_available():
        raise ModuleNotFoundError(
            "Reading labels.parquet/windows.parquet requires pyarrow or fastparquet."
        )
    input_dir = Path(input_dir)
    with np.load(input_dir / "features.npz") as arrays:
        X = arrays["X"]
        raw_windows = arrays["raw_windows"] if "raw_windows" in arrays else None
    feature_info = json.loads((input_dir / "feature_names.json").read_text(encoding="utf-8"))
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    pool = FeaturePool(
        X=X,
        labels=pd.read_parquet(input_dir / "labels.parquet"),
        windows=pd.read_parquet(input_dir / "windows.parquet"),
        feature_info=feature_info,
        manifest=manifest,
        raw_windows=raw_windows,
    )
    pool.validate()
    return pool


__all__ = [
    "FEATURE_INTERFACE_VERSION",
    "FEATURE_RECIPES",
    "FeaturePool",
    "build_grip_feature_pool",
    "load_feature_pool",
]
