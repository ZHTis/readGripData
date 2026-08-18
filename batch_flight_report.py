"""Execute the flight-analysis notebook for every paired run and collect outputs.

The source notebook remains the single analysis definition.  Each batch copy
changes only ``RUN_NAME`` before execution, so all runs follow the same cells,
filtering, event definitions, and plotting code.
"""

from __future__ import annotations

import base64
import argparse
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable

import nbformat
from nbclient import NotebookClient


RUN_ASSIGNMENT = re.compile(r'^RUN_NAME\s*=\s*["\'][^"\']+["\'].*$', re.MULTILINE)


def discover_run_names(data_dir: str | Path) -> list[str]:
    """Return run stems that have both ``RUN.dat`` and ``RUN_1.dat``."""
    root = Path(data_dir).expanduser().resolve()
    task_stems = {
        path.name[:-6]
        for path in root.rglob("*_1.dat")
        if path.is_file()
    }
    eeg_stems = {
        path.stem
        for path in root.rglob("*.dat")
        if path.is_file() and not path.name.endswith("_1.dat")
    }
    names = sorted(eeg_stems & task_stems, key=_natural_key)
    if not names:
        raise FileNotFoundError(f"No paired RUN.dat / RUN_1.dat files found under {root}")
    return names


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def _clear_execution(notebook: nbformat.NotebookNode) -> None:
    notebook.metadata.pop("widgets", None)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def _set_run_name(notebook: nbformat.NotebookNode, run_name: str) -> None:
    replacements = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        updated, count = RUN_ASSIGNMENT.subn(
            f'RUN_NAME = "{run_name}"  # batch-selected run', cell.source
        )
        if count:
            cell.source = updated
            replacements += count
    if replacements != 1:
        raise ValueError(
            f"Expected exactly one RUN_NAME assignment, found {replacements}"
        )


def _widget_images_for_cell(
    notebook: nbformat.NotebookNode,
    cell: nbformat.NotebookNode,
) -> list[str]:
    widget_bundle = (
        notebook.metadata.get("widgets", {})
        .get("application/vnd.jupyter.widget-state+json", {})
    )
    state = widget_bundle.get("state", {})
    roots: list[str] = []
    for output in cell.get("outputs", []):
        model = output.get("data", {}).get(
            "application/vnd.jupyter.widget-view+json", {}
        ).get("model_id")
        if model:
            roots.append(model)

    images: list[str] = []
    visited: set[str] = set()

    def visit(model_id: str) -> None:
        if model_id in visited or model_id not in state:
            return
        visited.add(model_id)
        model_state = state[model_id].get("state", {})
        for output in model_state.get("outputs", []):
            image_data = output.get("data", {}).get("image/png")
            if image_data:
                images.append(image_data)
        for child in model_state.get("children", []):
            visit(str(child).replace("IPY_MODEL_", ""))

    for root in roots:
        visit(root)
    return images


def _collect_cell_output(
    notebook: nbformat.NotebookNode,
    cell_index: int,
    asset_dir: Path,
) -> dict[str, Any]:
    cell = notebook.cells[cell_index]
    images: list[str] = []
    text_parts: list[str] = []
    html_parts: list[str] = []

    for output in cell.get("outputs", []):
        output_type = output.get("output_type")
        if output_type == "stream":
            text_parts.append(str(output.get("text", "")))
        elif output_type == "error":
            text_parts.append("\n".join(output.get("traceback", [])))
        else:
            data = output.get("data", {})
            if data.get("image/png"):
                images.append(data["image/png"])
            if data.get("text/html"):
                html_parts.append(str(data["text/html"]))
            elif data.get("text/plain") and not data.get(
                "application/vnd.jupyter.widget-view+json"
            ):
                text_parts.append(str(data["text/plain"]))

    images.extend(_widget_images_for_cell(notebook, cell))
    image_paths: list[str] = []
    for image_index, encoded in enumerate(images, 1):
        image_path = asset_dir / f"cell_{cell_index:02d}_{image_index:02d}.png"
        image_path.write_bytes(base64.b64decode(encoded))
        image_paths.append(str(image_path.resolve()))

    return {
        "cell_index": cell_index,
        "source_first_line": cell.source.strip().splitlines()[0]
        if cell.source.strip() else "",
        "images": image_paths,
        "text": "\n".join(text_parts).strip(),
        "html": "\n".join(html_parts).strip(),
    }


def execute_all_runs(
    source_notebook: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    run_names: Iterable[str] | None = None,
    timeout_s: int = 1200,
) -> dict[str, Any]:
    """Execute one notebook copy per run and save a comparison manifest."""
    source_path = Path(source_notebook).resolve()
    data_root = Path(data_dir).expanduser().resolve()
    report_root = Path(output_dir).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    selected_runs = list(run_names or discover_run_names(data_root))

    source = nbformat.read(source_path, as_version=4)
    code_indices = [
        index for index, cell in enumerate(source.cells)
        if cell.cell_type == "code"
    ]
    manifest: dict[str, Any] = {
        "source_notebook": str(source_path),
        "data_dir": str(data_root),
        "run_names": selected_runs,
        "code_cell_indices": code_indices,
        "runs": {},
    }

    for run_number, run_name in enumerate(selected_runs, 1):
        print(f"[{run_number}/{len(selected_runs)}] Executing {run_name} ...", flush=True)
        # Round-trip through nbformat so cell ``source`` fields are normalized
        # to strings rather than raw JSON line lists.
        notebook = nbformat.reads(nbformat.writes(source), as_version=4)
        _clear_execution(notebook)
        _set_run_name(notebook, run_name)
        client = NotebookClient(
            notebook,
            timeout=timeout_s,
            kernel_name="python3",
            resources={"metadata": {"path": str(source_path.parent)}},
            store_widget_state=True,
        )
        executed = client.execute()
        run_dir = report_root / run_name
        asset_dir = run_dir / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        executed_path = run_dir / f"{run_name}_flight_analysis.ipynb"
        nbformat.write(executed, executed_path)

        error_cells = [
            index for index, cell in enumerate(executed.cells)
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if error_cells:
            raise RuntimeError(f"{run_name} contains notebook errors in cells {error_cells}")
        manifest["runs"][run_name] = {
            "executed_notebook": str(executed_path.resolve()),
            "cells": {
                str(index): _collect_cell_output(executed, index, asset_dir)
                for index in code_indices
            },
        }

    manifest_path = report_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved manifest: {manifest_path}")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_title(manifest: dict[str, Any], cell_index: int) -> str:
    first_run = manifest["run_names"][0]
    first_line = manifest["runs"][first_run]["cells"][str(cell_index)][
        "source_first_line"
    ]
    return first_line.removeprefix("#").strip() or f"Code cell {cell_index}"


def comparison_html(
    manifest: dict[str, Any],
    cell_index: int,
    *,
    max_text_chars: int = 1800,
) -> str:
    """Build a five-panel HTML comparison for display in the batch notebook."""
    panels: list[str] = []
    for run_name in manifest["run_names"]:
        item = manifest["runs"][run_name]["cells"][str(cell_index)]
        media: list[str] = []
        for image_path in item["images"]:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            media.append(
                '<img src="data:image/png;base64,' + encoded
                + '" style="width:100%;height:auto;display:block;margin:0 0 6px 0;">'
            )
        if not media and item["html"]:
            media.append('<div class="batch-table">' + item["html"] + "</div>")
        text_value = item["text"]
        if len(text_value) > max_text_chars:
            text_value = text_value[:max_text_chars] + "\n…"
        if text_value:
            media.append("<pre>" + html.escape(text_value) + "</pre>")
        if not media:
            media.append('<div class="empty-output">No visible output</div>')
        panels.append(
            '<section class="run-panel"><h3>' + html.escape(run_name)
            + "</h3>" + "".join(media) + "</section>"
        )

    return """
    <style>
      .batch-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr));
                    gap:8px; align-items:start; width:100%; }
      .run-panel { min-width:0; border-top:3px solid #35618F; padding-top:4px; }
      .run-panel h3 { font:700 15px Arial,sans-serif; margin:0 0 5px; color:#252A31; }
      .run-panel pre { white-space:pre-wrap; overflow-wrap:anywhere; font:9px/1.25 Consolas,monospace;
                       margin:4px 0; max-height:440px; overflow:hidden; }
      .run-panel table { width:100%; border-collapse:collapse; font:8px Arial,sans-serif; }
      .run-panel th,.run-panel td { border-bottom:1px solid #ddd; padding:2px; }
      .empty-output { color:#777; font:italic 12px Arial,sans-serif; padding:16px 0; }
    </style>
    <div class="batch-grid">""" + "".join(panels) + "</div>"


__all__ = [
    "cell_title",
    "comparison_html",
    "discover_run_names",
    "execute_all_runs",
    "load_manifest",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute the single-run flight notebook for every paired run."
    )
    parser.add_argument("--source", default="align_R09_flight_eeg_force.ipynb")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/batch_flight_report")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    execute_all_runs(
        args.source,
        args.data_dir,
        args.output_dir,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    main()
