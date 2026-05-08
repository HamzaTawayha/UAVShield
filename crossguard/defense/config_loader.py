from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from crossguard.defense.invariants import InvariantConfig


def load_invariant_config(path: str | Path) -> InvariantConfig:
    """Load a flat YAML-like config file without requiring PyYAML."""

    path = Path(path)
    data = _load_with_pyyaml(path)
    if data is None:
        data = _load_flat_yaml(path)

    allowed = {field.name for field in fields(InvariantConfig)}
    kwargs = {key: value for key, value in data.items() if key in allowed}
    return InvariantConfig(**kwargs)


def _load_with_pyyaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return None

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    return loaded


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Cannot parse config line: {raw_line!r}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")
