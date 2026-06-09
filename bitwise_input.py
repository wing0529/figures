from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


EXP_RE = re.compile(
    r"^fault_(?P<model>[^_]+)_fp16_bitwise16_bit16-(?P<bit>\d+)_"
    r"(?:bersem-independent_)?lpddr5_Vendor-A_trefi(?P<trefi>\d+)_"
    r"tres(?P<tres>[0-9.]+)h_T(?P<temp>[-0-9.]+)_"
    r"ber-(?P<ber>[^_]+)_seed-(?P<seed>\d+)$"
)


def canonical_model(model: str) -> str:
    model = model.lower()
    return "resnet50" if model == "resnet" else model


def coerce_float(value: Any) -> float | None:
    if value in (None, "", "null", "None", "N/A"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text == "nan":
        return math.nan
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text in {"-inf", "-infinity"}:
        return -math.inf
    return float(text)


def payload_metric(payload: dict[str, Any], metric: str) -> float | None:
    value = coerce_float(payload.get(metric))
    if value is not None:
        return value
    harness = payload.get("harness_result", {})
    if isinstance(harness, dict):
        return coerce_float(harness.get(metric))
    return None


def _looks_like_csv(path: Path) -> bool:
    if path.suffix.lower() == ".csv":
        return True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.lstrip()
            if stripped:
                return not stripped.startswith("{")
    return False


def _paths(paths: Path | Iterable[Path]) -> list[Path]:
    if isinstance(paths, Path):
        return [paths]
    return list(paths)


def _iter_json_objects(line: str, path: Path, lineno: int) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(line)
    yielded = False
    while idx < length:
        while idx < length and line[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            payload, idx = decoder.raw_decode(line, idx)
        except json.JSONDecodeError as exc:
            next_obj = line.find("{", idx + 1)
            if yielded and next_obj < 0:
                break
            if next_obj >= 0:
                idx = next_obj
                continue
            raise ValueError(
                f"{path}:{lineno} is not valid JSONL. CSV inputs are auto-detected by .csv suffix."
            ) from exc
        if isinstance(payload, dict):
            yielded = True
            yield payload


def _load_csv_metric_rows(paths: list[Path], model: str, metric: str) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    want_model = canonical_model(model)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or metric not in reader.fieldnames:
                raise KeyError(f"Metric column {metric!r} not found in {path}")
            for row in reader:
                exp = str(row.get("experiment", ""))
                if exp:
                    exp_name = exp.removeprefix("fault_")
                    if not exp.startswith(f"fault_{want_model}_") and not exp_name.startswith(f"{want_model}_"):
                        continue
                bit = int(row["fault_param_val"])
                trefi = int(row["trefi_val"])
                rows_by_key[(trefi, bit)] = {
                    "experiment": exp,
                    "fault_param_val": bit,
                    "trefi_val": trefi,
                    "ber": float(row["ber"]),
                    "metric_value": coerce_float(row.get(metric)),
                    "seed": int(float(row.get("seed") or 0)),
                }
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def _load_jsonl_metric_rows(paths: list[Path], model: str, metric: str) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    want_model = canonical_model(model)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                for payload in _iter_json_objects(line, path, lineno):
                    exp = str(payload.get("experiment", ""))
                    match = EXP_RE.match(exp)
                    if not match or canonical_model(match.group("model")) != want_model:
                        continue
                    bit = int(match.group("bit"))
                    trefi = int(match.group("trefi"))
                    rows_by_key[(trefi, bit)] = {
                        "experiment": exp,
                        "fault_param_val": bit,
                        "trefi_val": trefi,
                        "ber": float(match.group("ber")),
                        "metric_value": payload_metric(payload, metric),
                        "seed": int(match.group("seed")),
                    }
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def load_metric_rows(paths: Path | Iterable[Path], model: str, metric: str) -> list[dict[str, Any]]:
    input_paths = _paths(paths)
    rows: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.exists():
            continue
        if _looks_like_csv(path):
            rows.extend(_load_csv_metric_rows([path], model, metric))
        else:
            rows.extend(_load_jsonl_metric_rows([path], model, metric))
    if not rows:
        searched = ", ".join(str(path) for path in input_paths)
        raise RuntimeError(f"No bitwise16 rows found for model={model}, metric={metric} in {searched}")
    rows_by_key = {(row["trefi_val"], row["fault_param_val"]): row for row in rows}
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def as_bar_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bit": row["fault_param_val"],
            "trefi": row["trefi_val"],
            "ber": row["ber"],
            "value": row["metric_value"],
        }
        for row in rows
    ]
