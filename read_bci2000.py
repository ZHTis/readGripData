"""Small, dependency-light reader for BCI2000 .dat files.

The signal array is exposed as a memory map, so even large recordings do not
need to be copied into RAM. State-vector fields are decoded on demand.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


TYPE_INFO = {
    "int16": ("<i2", 2),
    "int32": ("<i4", 4),
    "float32": ("<f4", 4),
}


class BCI2000Dat:
    def __init__(self, filename: str | Path):
        self.path = Path(filename)
        with self.path.open("rb") as stream:
            first_line = stream.readline().decode("latin1")
            self.header_len = self._header_number(first_line, "HeaderLen")
            self.source_channels = self._header_number(first_line, "SourceCh")
            self.statevector_len = self._header_number(first_line, "StatevectorLen")
            match = re.search(r"DataFormat=\s*(\w+)", first_line)
            if not match or match.group(1) not in TYPE_INFO:
                raise ValueError("Unsupported or missing BCI2000 DataFormat")
            self.data_format = match.group(1)
            stream.seek(0)
            self.header = stream.read(self.header_len).decode("latin1", "replace")

        self.signal_dtype, item_size = TYPE_INFO[self.data_format]
        self.record_size = self.source_channels * item_size + self.statevector_len
        payload = self.path.stat().st_size - self.header_len
        self.samples, remainder = divmod(payload, self.record_size)
        if remainder:
            raise ValueError(f"Truncated payload: {remainder} extra bytes")

        record_dtype = np.dtype([
            ("signal", self.signal_dtype, (self.source_channels,)),
            ("states", np.uint8, (self.statevector_len,)),
        ])
        self._records = np.memmap(
            self.path,
            mode="r",
            dtype=record_dtype,
            offset=self.header_len,
            shape=(self.samples,),
        )
        self.signals = self._records["signal"]
        self._state_bytes = self._records["states"]
        self.state_definitions = self._parse_state_definitions()
        self.sampling_rate = self._parameter_number("SamplingRate")
        self.channel_names = self._channel_names()

    @staticmethod
    def _header_number(text: str, name: str) -> int:
        match = re.search(rf"\b{re.escape(name)}=\s*(\d+)", text)
        if not match:
            raise ValueError(f"Missing {name}")
        return int(match.group(1))

    def _parameter_number(self, name: str) -> float:
        match = re.search(
            rf"^.*\b{re.escape(name)}=\s*([-+]?\d+(?:\.\d+)?)",
            self.header,
            re.MULTILINE,
        )
        if not match:
            raise ValueError(f"Missing parameter {name}")
        return float(match.group(1))

    def _channel_names(self) -> list[str]:
        match = re.search(r"^.*\bChannelNames=\s*(.*?)\s*//", self.header, re.MULTILINE)
        if not match:
            return [f"Channel{i + 1}" for i in range(self.source_channels)]
        fields = match.group(1).split()
        count = int(fields[0])
        return fields[1 : count + 1]

    def _parse_state_definitions(self) -> dict[str, tuple[int, int, int]]:
        section = self.header.split("[ State Vector Definition ]", 1)[1]
        section = section.split("[ Parameter Definition ]", 1)[0]
        definitions = {}
        for line in section.splitlines():
            match = re.match(r"^(\S+)\s+(\d+)\s+\d+\s+(\d+)\s+(\d+)", line)
            if match:
                name, bits, byte_offset, bit_offset = match.groups()
                definitions[name] = (int(bits), int(byte_offset), int(bit_offset))
        return definitions

    def state(self, name: str) -> np.ndarray:
        bits, byte_offset, bit_offset = self.state_definitions[name]
        if bits > 64:
            raise ValueError("State fields wider than 64 bits are unsupported")
        result = np.zeros(self.samples, dtype=np.uint64)
        for bit_index in range(bits):
            absolute_bit = bit_offset + bit_index
            byte = self._state_bytes[:, byte_offset + absolute_bit // 8]
            value = ((byte >> (absolute_bit % 8)) & 1).astype(np.uint64)
            result |= value << bit_index
        return result


def export_task_csv(recording: BCI2000Dat, output_dir: Path) -> tuple[Path, Path]:
    stem = recording.path.stem
    timeseries_path = output_dir / f"{stem}_task_timeseries.csv"
    trials_path = output_dir / f"{stem}_trials.csv"
    wanted = [
        "SessionTrialIndex", "BlockIndex", "BlockTrialIndex", "BlockPhase",
        "TargetCode", "Feedback", "TrialResult", "GripMarker",
        "GripForceRaw", "GripForceNormalized", "CursorPosY",
    ]
    states = {name: recording.state(name) for name in wanted if name in recording.state_definitions}

    with timeseries_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "time_s", *recording.channel_names, *states])
        for index in range(recording.samples):
            writer.writerow(
                [index, index / recording.sampling_rate]
                + recording.signals[index].tolist()
                + [int(values[index]) for values in states.values()]
            )

    trial_index = states.get("SessionTrialIndex")
    feedback = states.get("Feedback")
    result = states.get("TrialResult")
    normalized = states.get("GripForceNormalized")
    with trials_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trial", "start_s", "end_s", "feedback_start_s", "feedback_end_s",
            "success", "completed", "max_grip_signal", "max_normalized_force",
        ])
        for trial in range(1, int(trial_index.max()) + 1):
            indices = np.flatnonzero(trial_index == trial)
            fb_indices = indices[feedback[indices] > 0]
            completed = bool(np.any(result[indices] > 0))
            writer.writerow([
                trial,
                indices[0] / recording.sampling_rate,
                indices[-1] / recording.sampling_rate,
                fb_indices[0] / recording.sampling_rate if fb_indices.size else "",
                fb_indices[-1] / recording.sampling_rate if fb_indices.size else "",
                int(completed),
                int(completed),
                float(np.max(recording.signals[indices, 0])),
                float(np.max(normalized[indices]) / 65535) if normalized is not None else "",
            ])
    return timeseries_path, trials_path


def export_flight_task_csv(recording: BCI2000Dat, output_dir: Path) -> tuple[Path, Path]:
    stem = recording.path.stem
    timeseries_path = output_dir / f"{stem}_flight_timeseries.csv"
    trials_path = output_dir / f"{stem}_flight_trials.csv"
    wanted = [
        "GamePhase", "TargetCode", "Feedback", "ResultCode",
        "BallWorldX", "BallWorldY", "BallVelocityY", "CameraWorldX",
        "GripForceRaw", "GripForceNormalized", "Collision",
        "CollisionObject", "FlightTrialResult",
    ]
    states = {name: recording.state(name) for name in wanted if name in recording.state_definitions}
    game_phase = states["GamePhase"]
    trial_starts = np.flatnonzero((game_phase == 1) & np.r_[True, game_phase[:-1] != 1])
    trial_number = np.zeros(recording.samples, dtype=np.uint32)
    for number, start in enumerate(trial_starts, 1):
        end = trial_starts[number] if number < len(trial_starts) else recording.samples
        trial_number[start:end] = number

    with timeseries_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "time_s", "trial", *recording.channel_names, *states])
        for index in range(recording.samples):
            writer.writerow(
                [index, index / recording.sampling_rate, int(trial_number[index])]
                + recording.signals[index].tolist()
                + [int(values[index]) for values in states.values()]
            )

    result = states["FlightTrialResult"]
    normalized = states["GripForceNormalized"]
    with trials_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "trial", "start_s", "end_s", "flight_start_s", "flight_end_s",
            "result_code", "result", "collision", "max_grip_signal",
            "max_normalized_force",
        ])
        for number in range(1, len(trial_starts) + 1):
            indices = np.flatnonzero(trial_number == number)
            flight = indices[game_phase[indices] == 2]
            result_code = int(np.max(result[indices]))
            result_label = {0: "incomplete", 1: "success", 2: "failure"}.get(
                result_code, f"code_{result_code}"
            )
            writer.writerow([
                number,
                indices[0] / recording.sampling_rate,
                indices[-1] / recording.sampling_rate,
                flight[0] / recording.sampling_rate if flight.size else "",
                flight[-1] / recording.sampling_rate if flight.size else "",
                result_code,
                result_label,
                int(np.any(states["Collision"][indices] > 0)),
                float(np.max(recording.signals[indices, 0])),
                float(np.max(normalized[indices]) / 65535),
            ])
    return timeseries_path, trials_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--export-task", action="store_true")
    args = parser.parse_args()
    for filename in args.files:
        recording = BCI2000Dat(filename)
        print(
            f"{filename.name}: {recording.samples} samples, "
            f"{recording.source_channels} channels, {recording.sampling_rate:g} Hz, "
            f"{recording.samples / recording.sampling_rate:.3f} s"
        )
        print("  channels:", ", ".join(recording.channel_names))
        print("  states:", ", ".join(recording.state_definitions))
        if args.export_task and "SessionTrialIndex" in recording.state_definitions:
            for path in export_task_csv(recording, filename.parent):
                print("  wrote:", path)
        elif args.export_task and "GamePhase" in recording.state_definitions:
            for path in export_flight_task_csv(recording, filename.parent):
                print("  wrote:", path)


if __name__ == "__main__":
    main()
