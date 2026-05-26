from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler


DEFAULT_TRAIN_SEEDS = [42, 123, 256, 512, 1024, 2048]
DEFAULT_VAL_SEEDS = [4096, 8192]
DEFAULT_TEST_SEEDS = [111, 222]
LEAKAGE_FIELDS = {
    "attack",
    "injected_payload",
    "false_belief_label",
    "unsafe_tool_label",
    "termination_reason",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate agent-log attack detectors from Ibrahim's Results/events.jsonl logs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("../Results"))
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help=(
            "Optional aggregate metrics.csv. When provided, rows in this file become the "
            "dataset source of truth, so S1-S4 can be used even if some events.jsonl files "
            "are not present locally."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/agent_log_detector"))
    parser.add_argument(
        "--split-strategy",
        choices=["seed", "random"],
        default="seed",
        help="Use seed-heldout split or shuffled stratified run-level split.",
    )
    parser.add_argument("--train-size", type=float, default=0.80)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--train-seeds", type=parse_seed_list, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--val-seeds", type=parse_seed_list, default=DEFAULT_VAL_SEEDS)
    parser.add_argument("--test-seeds", type=parse_seed_list, default=DEFAULT_TEST_SEEDS)
    parser.add_argument(
        "--require-events",
        action="store_true",
        help="Fail if --metrics-csv contains a run that does not have a matching events.jsonl.",
    )
    parser.add_argument(
        "--ignore-events",
        action="store_true",
        help="Use only --metrics-csv rows and ignore events.jsonl, useful when only aggregate CSV data is available.",
    )
    parser.add_argument(
        "--supervised-model",
        choices=["logistic_regression", "random_forest", "extra_trees"],
        default="logistic_regression",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "accuracy"],
        default="f1",
        help="Validation metric used to select the supervised detector threshold.",
    )
    parser.add_argument(
        "--threshold-scope",
        choices=["global", "scenario", "memory_mode", "scenario_memory"],
        default="global",
        help="Use one global threshold or validation-calibrated thresholds by known evaluation context.",
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    runs = load_runs(
        args.results_dir,
        metrics_csv=args.metrics_csv,
        require_events=args.require_events,
        ignore_events=args.ignore_events,
    )
    if not runs:
        raise SystemExit(f"No events.jsonl logs found under {args.results_dir}")

    if args.split_strategy == "random":
        train_runs, val_runs, test_runs = split_random_stratified(
            runs,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        train_runs, val_runs, test_runs = split_by_seed(
            runs,
            train_seeds=set(args.train_seeds),
            val_seeds=set(args.val_seeds),
            test_seeds=set(args.test_seeds),
        )
    validate_split(train_runs, val_runs, test_runs)

    print_dataset_summary(runs, train_runs, val_runs, test_runs)

    train_artifacts = fit_feature_builder(train_runs)
    x_train = transform_runs(train_artifacts, train_runs)
    x_val = transform_runs(train_artifacts, val_runs)
    x_test = transform_runs(train_artifacts, test_runs)
    y_train = labels(train_runs)
    y_val = labels(val_runs)

    supervised = build_supervised_model(args.supervised_model, args.seed)
    supervised.fit(x_train, y_train)
    supervised_val_scores = supervised_scores(supervised, x_val)
    supervised_test_scores = supervised_scores(supervised, x_test)
    supervised_threshold = calibrate_thresholds(
        runs=val_runs,
        scores=supervised_val_scores,
        y=y_val,
        metric=args.threshold_metric,
        scope=args.threshold_scope,
    )
    supervised_eval = evaluate_detector(
        detector_name=f"supervised_{args.supervised_model}",
        runs=test_runs,
        scores=supervised_test_scores,
        threshold=supervised_threshold,
        threshold_scope=args.threshold_scope,
    )
    supervised_eval["validation"] = evaluate_detector(
        detector_name=f"supervised_{args.supervised_model}",
        runs=val_runs,
        scores=supervised_val_scores,
        threshold=supervised_threshold,
        threshold_scope=args.threshold_scope,
        include_predictions=False,
    )["test"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, supervised_eval)
    write_feature_table(output_dir / "agent_log_features.csv", runs)
    write_top_features(output_dir / "supervised_top_features.csv", supervised, train_artifacts)
    write_plots(output_dir, supervised_eval)
    joblib.dump(
        {
            "supervised_model": supervised,
            "feature_builder": train_artifacts,
            "supervised_threshold": supervised_threshold,
            "leakage_fields_excluded": sorted(LEAKAGE_FIELDS),
        },
        output_dir / "agent_log_detectors.joblib",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "results_dir": str(args.results_dir),
                "metrics_csv": str(args.metrics_csv) if args.metrics_csv else None,
                "ignore_events": args.ignore_events,
                "require_events": args.require_events,
                "output_dir": str(output_dir),
                "train_seeds": args.train_seeds,
                "val_seeds": args.val_seeds,
                "test_seeds": args.test_seeds,
                "split_strategy": args.split_strategy,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "supervised_model": args.supervised_model,
                "threshold_metric": args.threshold_metric,
                "threshold_scope": args.threshold_scope,
                "seed": args.seed,
                "leakage_fields_excluded": sorted(LEAKAGE_FIELDS),
                "feature_families": {
                    "numeric": "event counts, tool calls, memory writes, safety status, observation deltas, telemetry deltas",
                    "metrics_csv": "safe aggregate runtime columns from metrics.csv when provided",
                    "text": "LLM rationale and non-oracle planner text only",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nWrote outputs to", output_dir)
    print_result(supervised_eval)


def parse_seed_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return value
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_runs(
    results_dir: Path,
    metrics_csv: Path | None = None,
    require_events: bool = False,
    ignore_events: bool = False,
) -> list[dict[str, Any]]:
    event_index = {} if ignore_events else index_event_logs(results_dir)
    if metrics_csv is not None:
        return load_runs_from_metrics_csv(metrics_csv, event_index, require_events)

    runs: list[dict[str, Any]] = []
    for path in sorted(event_index.values()):
        events = read_jsonl(path)
        if not events:
            continue
        meta = extract_metadata(events[0], path)
        features, text = extract_features(events)
        summary_path = path.with_name("summary.json")
        summary = read_json(summary_path) if summary_path.exists() else {}
        runs.append(
            {
                "path": str(path),
                "summary_path": str(summary_path) if summary_path.exists() else "",
                "run_id": meta["run_id"],
                "seed": int(meta["seed"]),
                "model": str(meta["model"]),
                "scenario": str(meta["scenario"]),
                "condition": str(meta["condition"]),
                "memory_mode": str(meta["memory_mode"]),
                "label": int(str(meta["condition"]) != "clean"),
                "attack_family": attack_family(str(meta["condition"]), str(meta["scenario"])),
                "features": features,
                "text": text,
                "mission_failure": int(summary.get("mission_failure", 0) or 0),
            }
        )
    return runs


def index_event_logs(results_dir: Path) -> dict[tuple[str, str, str, str, int, str], Path]:
    event_index: dict[tuple[str, str, str, str, int, str], Path] = {}
    for path in sorted(results_dir.glob("**/events.jsonl")):
        events = read_jsonl(path)
        if not events:
            continue
        meta = extract_metadata(events[0], path)
        event_index[event_key(meta)] = path
    return event_index


def load_runs_from_metrics_csv(
    metrics_csv: Path,
    event_index: dict[tuple[str, str, str, str, int, str], Path],
    require_events: bool,
) -> list[dict[str, Any]]:
    rows = read_csv(metrics_csv)
    runs: list[dict[str, Any]] = []
    missing_event_count = 0
    for row in rows:
        meta = {
            "run_id": row["run_id"],
            "seed": int(row["seed"]),
            "model": row["model"],
            "scenario": row["scenario"],
            "memory_mode": row["memory_mode"],
            "condition": row["condition"],
        }
        path = event_index.get(event_key(meta))
        if path is None:
            missing_event_count += 1
            if require_events:
                raise SystemExit(
                    "Missing events.jsonl for metrics row "
                    f"scenario={row['scenario']} condition={row['condition']} "
                    f"model={row['model']} memory_mode={row['memory_mode']} seed={row['seed']}"
                )
            features: dict[str, float | str] = {"event_log_available": 0.0}
            text = ""
            summary: dict[str, Any] = {}
            summary_path = ""
            path_text = ""
        else:
            events = read_jsonl(path)
            features, text = extract_features(events)
            features["event_log_available"] = 1.0
            summary_path_obj = path.with_name("summary.json")
            summary = read_json(summary_path_obj) if summary_path_obj.exists() else {}
            summary_path = str(summary_path_obj) if summary_path_obj.exists() else ""
            path_text = str(path)

        features.update(extract_metrics_features(row))
        runs.append(
            {
                "path": path_text,
                "summary_path": summary_path,
                "run_id": meta["run_id"],
                "seed": meta["seed"],
                "model": str(meta["model"]),
                "scenario": str(meta["scenario"]),
                "condition": str(meta["condition"]),
                "memory_mode": str(meta["memory_mode"]),
                "label": int(str(meta["condition"]) != "clean"),
                "attack_family": attack_family(str(meta["condition"]), str(meta["scenario"])),
                "features": features,
                "text": text,
                "mission_failure": int(to_bool(row.get("mission_failure")) or summary.get("mission_failure", 0) or 0),
            }
        )
    print(
        f"Loaded {len(rows)} rows from {metrics_csv}; "
        f"matched events.jsonl for {len(rows) - missing_event_count}, "
        f"metrics-only rows for {missing_event_count}."
    )
    return runs


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def event_key(meta: dict[str, Any]) -> tuple[str, str, str, str, int, str]:
    return (
        str(meta["scenario"]),
        str(meta["condition"]),
        str(meta["memory_mode"]),
        str(meta["model"]),
        int(meta["seed"]),
        str(meta["run_id"]),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return events


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics_features(row: dict[str, str]) -> dict[str, float | str]:
    """Use only aggregate columns that could plausibly be known from a completed run.

    We deliberately skip oracle/result labels such as fbar/rmfr/uter, ghost counts,
    attack condition, defense metrics, and ground-truth distance-to-person fields.
    """

    numeric_columns = [
        "steps",
        "elapsed_s",
        "total_distance_m",
        "memory_citation_count",
        "total_llm_calls",
        "parse_success_rate",
        "llm_latency_mean_sec",
        "llm_latency_p95_sec",
        "llm_latency_total_sec",
    ]
    features: dict[str, float | str] = {}
    for column in numeric_columns:
        value = as_float(row.get(column))
        if not math.isnan(value):
            features[f"metrics_{column}"] = value
    for column in ["scenario", "memory_mode", "planner_type", "current_safety_status"]:
        value = row.get(column)
        if value:
            features[f"metrics_category_{column}"] = value
    return features


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def extract_metadata(first_event: dict[str, Any], path: Path) -> dict[str, Any]:
    required = ["run_id", "seed", "model", "scenario", "memory_mode", "condition"]
    missing = [field for field in required if field not in first_event]
    if missing:
        raise SystemExit(f"{path} is missing metadata fields: {', '.join(missing)}")
    return {field: first_event[field] for field in required}


def attack_family(condition: str, scenario: str) -> str:
    if condition == "clean":
        return "clean"
    if condition.endswith("_attack"):
        return condition.removesuffix("_attack")
    return scenario


def extract_features(events: list[dict[str, Any]]) -> tuple[dict[str, float | str], str]:
    numeric: defaultdict[str, float] = defaultdict(float)
    categories: dict[str, str] = {}
    text_parts: list[str] = []
    tool_counts: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    safety_states: Counter[str] = Counter()
    memory_types: Counter[str] = Counter()
    memory_keys: Counter[str] = Counter()
    battery_values: list[float] = []
    speed_values: list[float] = []
    alt_values: list[float] = []
    true_visible_deltas: list[float] = []
    visible_detection_counts: list[float] = []
    true_detection_counts: list[float] = []
    max_memory_size = 0
    max_memory_importance = 0.0
    cited_counts: list[float] = []
    llm_latency: list[float] = []

    for event in events:
        event_type = str(event.get("type", "unknown"))
        numeric["events_total"] += 1
        numeric[f"events_type_{event_type}"] += 1

        step = as_float(event.get("step"))
        if not math.isnan(step):
            numeric["max_step"] = max(numeric["max_step"], step)

        tool_name = str(event.get("tool_call_name") or nested_get(event, ["tool_result", "tool_name"]) or "none")
        tool_counts[tool_name] += 1

        if event_type == "decision":
            numeric["decision_events"] += 1
            numeric["repair_used_count"] += int(bool(event.get("repair_used")))
            numeric["parse_failure_count"] += int(event.get("parse_success") is False)
            numeric["validation_failure_count"] += int(event.get("validation_success") is False)
            numeric["retry_count_sum"] += as_float(event.get("retry_count"), default=0.0)
            cited = event.get("cited_memory_ids") or []
            if isinstance(cited, list):
                cited_counts.append(float(len(cited)))
                numeric["cited_memory_total"] += len(cited)
            raw_call = event.get("raw_llm_tool_call")
            if isinstance(raw_call, dict):
                safety_states[str(raw_call.get("safety_status", "missing"))] += 1
            text_parts.append(str(event.get("rationale") or ""))
            text_parts.append(str(nested_get(event, ["raw_llm_tool_call", "rationale"]) or ""))
            text_parts.append(str(event.get("repair_type") or ""))
            text_parts.append(str(event.get("repair_reason") or ""))
            latency = nested_get(event, ["llm", "llm_latency_sec"])
            if latency is None:
                latency = event.get("llm_latency_sec")
            latency_float = as_float(latency)
            if not math.isnan(latency_float):
                llm_latency.append(latency_float)

        if event_type == "observation":
            numeric["observation_events"] += 1
            true_observation = event.get("true_observation")
            visible_observation = event.get("agent_visible_observation")
            true_visible_deltas.append(json_delta_size(true_observation, visible_observation))
            true_detections = count_detections(true_observation)
            visible_detections = count_detections(visible_observation)
            true_detection_counts.append(true_detections)
            visible_detection_counts.append(visible_detections)
            numeric["visible_minus_true_detections_sum"] += visible_detections - true_detections
            safety_value = extract_visible_safety(visible_observation)
            if safety_value:
                safety_states[safety_value] += 1
            text_parts.append(str(nested_get(event, ["tool_result", "message"]) or ""))
            text_parts.append(str(nested_get(event, ["executor_result", "message"]) or ""))

        telemetry = event.get("telemetry_after") or event.get("telemetry_before")
        if isinstance(telemetry, dict):
            phases[str(telemetry.get("mission_phase", "missing"))] += 1
            battery = as_float(telemetry.get("battery_remaining_pct"))
            if not math.isnan(battery):
                battery_values.append(battery)
            speed = as_float(telemetry.get("groundspeed_m_s"))
            if not math.isnan(speed):
                speed_values.append(speed)
            alt = as_float(nested_get(telemetry, ["position", "rel_alt_m"]))
            if not math.isnan(alt):
                alt_values.append(alt)
            numeric["health_not_ok_count"] += int(telemetry.get("health_all_ok") is False)

        for field in ("memory_before", "memory_write_candidate", "memory_written", "memory_after"):
            value = event.get(field)
            if isinstance(value, list):
                numeric[f"{field}_total"] += len(value)
                if field == "memory_after":
                    max_memory_size = max(max_memory_size, len(value))
                if field == "memory_written":
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        memory_types[str(item.get("type", "missing"))] += 1
                        memory_keys[str(item.get("key", "missing"))] += 1
                        importance = as_float(item.get("importance"))
                        if not math.isnan(importance):
                            max_memory_importance = max(max_memory_importance, importance)
                        text_parts.append(str(item.get("type") or ""))
                        text_parts.append(str(item.get("key") or ""))

    for tool, count in tool_counts.items():
        numeric[f"tool_count_{clean_name(tool)}"] = float(count)
    for phase, count in phases.items():
        numeric[f"phase_count_{clean_name(phase)}"] = float(count)
    for state, count in safety_states.items():
        numeric[f"safety_state_count_{clean_name(state)}"] = float(count)
    for memory_type, count in memory_types.items():
        numeric[f"memory_type_count_{clean_name(memory_type)}"] = float(count)
    for memory_key, count in memory_keys.items():
        numeric[f"memory_key_count_{clean_name(memory_key)}"] = float(count)

    add_stats(numeric, "battery", battery_values)
    add_stats(numeric, "speed", speed_values)
    add_stats(numeric, "altitude", alt_values)
    add_stats(numeric, "true_visible_delta", true_visible_deltas)
    add_stats(numeric, "true_detection_count", true_detection_counts)
    add_stats(numeric, "visible_detection_count", visible_detection_counts)
    add_stats(numeric, "cited_memory_count", cited_counts)
    add_stats(numeric, "llm_latency_sec", llm_latency)
    numeric["max_memory_size"] = float(max_memory_size)
    numeric["max_memory_importance"] = float(max_memory_importance)
    numeric["tool_diversity"] = float(len(tool_counts))
    numeric["phase_diversity"] = float(len(phases))
    numeric["memory_type_diversity"] = float(len(memory_types))

    if numeric["events_total"]:
        numeric["repair_rate"] = numeric["repair_used_count"] / numeric["events_total"]
        numeric["validation_failure_rate"] = numeric["validation_failure_count"] / numeric["events_total"]
        numeric["parse_failure_rate"] = numeric["parse_failure_count"] / numeric["events_total"]
        numeric["memory_written_per_event"] = numeric["memory_written_total"] / numeric["events_total"]
        numeric["cited_memory_per_event"] = numeric["cited_memory_total"] / numeric["events_total"]

    categories["memory_mode"] = str(events[0].get("memory_mode", "unknown"))
    features: dict[str, float | str] = dict(numeric)
    features.update({f"category_{key}": value for key, value in categories.items()})
    text = " ".join(part for part in text_parts if part).lower()
    return features, text


def nested_get(value: Any, path: list[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def json_delta_size(left: Any, right: Any) -> float:
    left_text = json.dumps(strip_oracle_fields(left), sort_keys=True, separators=(",", ":"))
    right_text = json.dumps(strip_oracle_fields(right), sort_keys=True, separators=(",", ":"))
    if left_text == right_text:
        return 0.0
    return float(abs(len(left_text) - len(right_text)) + 1)


def strip_oracle_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_oracle_fields(item) for key, item in value.items() if key not in LEAKAGE_FIELDS}
    if isinstance(value, list):
        return [strip_oracle_fields(item) for item in value]
    return value


def count_detections(observation: Any) -> float:
    if not isinstance(observation, dict):
        return 0.0
    detections = nested_get(observation, ["data", "detections"])
    if isinstance(detections, list):
        return float(len(detections))
    detections = observation.get("detections")
    if isinstance(detections, list):
        return float(len(detections))
    return 0.0


def extract_visible_safety(observation: Any) -> str:
    if not isinstance(observation, dict):
        return ""
    safety = nested_get(observation, ["data", "safety_status"])
    if isinstance(safety, dict):
        return str(safety.get("state") or safety.get("constraint") or "structured")
    if safety:
        return str(safety)
    return ""


def add_stats(features: defaultdict[str, float], prefix: str, values: list[float]) -> None:
    if not values:
        return
    array = np.asarray(values, dtype=np.float64)
    features[f"{prefix}_mean"] = float(np.mean(array))
    features[f"{prefix}_std"] = float(np.std(array))
    features[f"{prefix}_min"] = float(np.min(array))
    features[f"{prefix}_max"] = float(np.max(array))
    features[f"{prefix}_range"] = float(np.max(array) - np.min(array))
    features[f"{prefix}_last_minus_first"] = float(array[-1] - array[0])


def clean_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "empty"


def split_by_seed(
    runs: list[dict[str, Any]],
    train_seeds: set[int],
    val_seeds: set[int],
    test_seeds: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train = [run for run in runs if run["seed"] in train_seeds]
    val = [run for run in runs if run["seed"] in val_seeds]
    test = [run for run in runs if run["seed"] in test_seeds]
    used = train_seeds | val_seeds | test_seeds
    unused = sorted({run["seed"] for run in runs} - used)
    if unused:
        print(f"Warning: unused seeds: {unused}")
    return train, val, test


def split_random_stratified(
    runs: list[dict[str, Any]],
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise SystemExit(
            f"train-size + val-size + test-size must equal 1.0, got {total:.4f}"
        )

    indices = np.arange(len(runs))
    strata = stratification_keys(runs)
    train_idx, holdout_idx = safe_train_test_split(
        indices,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=strata,
    )
    holdout_strata = strata[holdout_idx]
    relative_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = safe_train_test_split(
        holdout_idx,
        test_size=relative_test_size,
        random_state=seed + 1,
        stratify=holdout_strata,
    )
    return select_runs(runs, train_idx), select_runs(runs, val_idx), select_runs(runs, test_idx)


def safe_train_test_split(
    indices: np.ndarray,
    test_size: float,
    random_state: int,
    stratify: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )


def stratification_keys(runs: list[dict[str, Any]]) -> np.ndarray:
    keys = []
    for run in runs:
        event_available = int(float(run["features"].get("event_log_available", 1.0)) > 0.5)
        keys.append(
            "|".join(
                [
                    str(run["condition"]),
                    str(run["scenario"]),
                    str(run["memory_mode"]),
                    f"events{event_available}",
                ]
            )
        )
    return np.asarray(keys)


def select_runs(runs: list[dict[str, Any]], indices: np.ndarray) -> list[dict[str, Any]]:
    return [runs[int(index)] for index in indices]


def validate_split(train: list[dict[str, Any]], val: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    for name, split in (("train", train), ("validation", val), ("test", test)):
        if not split:
            raise SystemExit(f"{name} split is empty.")
        y = labels(split)
        if len(set(y.tolist())) < 2 and name != "train":
            raise SystemExit(f"{name} split needs both clean and attack runs.")
    if len(set(labels(train).tolist())) < 2:
        raise SystemExit("train split needs both clean and attack runs for the supervised detector.")


def fit_feature_builder(runs: list[dict[str, Any]]) -> dict[str, Any]:
    dict_vectorizer = DictVectorizer(sparse=True)
    x_dict = dict_vectorizer.fit_transform([run["features"] for run in runs])
    scaler = MaxAbsScaler()
    scaler.fit(x_dict)
    texts = [run["text"] for run in runs]
    text_vectorizer = None
    if any(text.strip() for text in texts):
        candidate = TfidfVectorizer(
            min_df=2,
            max_features=2000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z_]{2,}\b",
        )
        try:
            candidate.fit(texts)
            text_vectorizer = candidate
        except ValueError:
            text_vectorizer = None
    return {
        "dict_vectorizer": dict_vectorizer,
        "scaler": scaler,
        "text_vectorizer": text_vectorizer,
    }


def transform_runs(artifacts: dict[str, Any], runs: list[dict[str, Any]]):
    x_dict = artifacts["dict_vectorizer"].transform([run["features"] for run in runs])
    x_dict = artifacts["scaler"].transform(x_dict)
    if artifacts["text_vectorizer"] is None:
        return x_dict
    x_text = artifacts["text_vectorizer"].transform([run["text"] for run in runs])
    return hstack([x_dict, x_text], format="csr")


def feature_names(artifacts: dict[str, Any]) -> list[str]:
    dict_names = [f"log.{name}" for name in artifacts["dict_vectorizer"].get_feature_names_out()]
    if artifacts["text_vectorizer"] is None:
        text_names = []
    else:
        text_names = [f"text.{name}" for name in artifacts["text_vectorizer"].get_feature_names_out()]
    return [*dict_names, *text_names]


def labels(runs: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([run["label"] for run in runs], dtype=np.int64)


def build_supervised_model(name: str, seed: int):
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=1000,
            class_weight="balanced",
            min_samples_leaf=1,
            max_features=0.35,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )


def supervised_scores(model: Any, x: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    scores = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def best_threshold(scores: np.ndarray, y: np.ndarray, metric: str) -> float:
    best_value = -1.0
    best_recall = -1.0
    best_precision = -1.0
    best = 0.5
    for threshold in threshold_candidates(scores):
        pred = (scores >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        value = f1 if metric == "f1" else accuracy_score(y, pred)
        key = (value, recall, precision)
        if key > (best_value, best_recall, best_precision):
            best_value = value
            best_recall = float(recall)
            best_precision = float(precision)
            best = float(threshold)
    return best


def calibrate_thresholds(
    runs: list[dict[str, Any]],
    scores: np.ndarray,
    y: np.ndarray,
    metric: str,
    scope: str,
) -> float | dict[str, float]:
    if scope == "global":
        return best_threshold(scores, y, metric=metric)

    thresholds: dict[str, float] = {"__default__": best_threshold(scores, y, metric=metric)}
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, run in enumerate(runs):
        groups[threshold_group_key(run, scope)].append(index)

    for key, indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=np.int64)
        if len(set(y[idx].tolist())) < 2:
            thresholds[key] = thresholds["__default__"]
        else:
            thresholds[key] = best_threshold(scores[idx], y[idx], metric=metric)
    return thresholds


def threshold_group_key(run: dict[str, Any], scope: str) -> str:
    if scope == "scenario":
        return str(run["scenario"])
    if scope == "memory_mode":
        return str(run["memory_mode"])
    if scope == "scenario_memory":
        return f"{run['scenario']}|{run['memory_mode']}"
    return "__default__"


def threshold_values_for_runs(
    runs: list[dict[str, Any]],
    threshold: float | dict[str, float],
    scope: str,
) -> np.ndarray:
    if not isinstance(threshold, dict):
        return np.full(len(runs), float(threshold), dtype=np.float64)
    default = float(threshold.get("__default__", 0.5))
    return np.asarray(
        [float(threshold.get(threshold_group_key(run, scope), default)) for run in runs],
        dtype=np.float64,
    )


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if len(unique) <= 200:
        return unique
    return np.quantile(scores, np.linspace(0.01, 0.99, 199))


def evaluate_detector(
    detector_name: str,
    runs: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float | dict[str, float],
    threshold_scope: str = "global",
    include_predictions: bool = True,
) -> dict[str, Any]:
    y_true = labels(runs)
    threshold_values = threshold_values_for_runs(runs, threshold, threshold_scope)
    y_pred = (scores >= threshold_values).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    test = {
        "runs": int(len(runs)),
        "clean_runs": int((y_true == 0).sum()),
        "attack_runs": int((y_true == 1).sum()),
        "threshold": float(threshold) if not isinstance(threshold, dict) else None,
        "threshold_scope": threshold_scope,
        "thresholds": threshold if isinstance(threshold, dict) else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": safe_roc_auc(y_true, scores),
        "pr_auc": safe_pr_auc(y_true, scores),
        "fpr_at_tpr_0_90": fpr_at_tpr(y_true, scores, target_tpr=0.90),
        "normal_flag_rate": float(y_pred[y_true == 0].mean()) if (y_true == 0).any() else 0.0,
        "attack_flag_rate": float(y_pred[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
    }
    evaluation: dict[str, Any] = {
        "model": detector_name,
        "test": test,
        "by_attack_family": by_group(runs, scores, y_pred, "attack_family"),
        "by_scenario": by_group(runs, scores, y_pred, "scenario"),
        "by_memory_mode": by_group(runs, scores, y_pred, "memory_mode"),
    }
    if include_predictions:
        evaluation["predictions"] = prediction_rows(runs, scores, y_pred, threshold_values)
    return evaluation


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= target_tpr
    if not mask.any():
        return None
    return float(np.min(fpr[mask]))


def by_group(
    runs: list[dict[str, Any]],
    scores: np.ndarray,
    y_pred: np.ndarray,
    field: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, run in enumerate(runs):
        grouped[str(run[field])].append(index)
    out: dict[str, dict[str, float | int]] = {}
    y_true = labels(runs)
    for group, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=np.int64)
        group_true = y_true[idx]
        group_pred = y_pred[idx]
        out[group] = {
            "runs": int(len(idx)),
            "clean_runs": int((group_true == 0).sum()),
            "attack_runs": int((group_true == 1).sum()),
            "flagged_rate": float(group_pred.mean()),
            "mean_score": float(np.mean(scores[idx])),
        }
        if (group_true == 1).any():
            out[group]["attack_recall"] = float(group_pred[group_true == 1].mean())
        if (group_true == 0).any():
            out[group]["false_positive_rate"] = float(group_pred[group_true == 0].mean())
    return out


def prediction_rows(
    runs: list[dict[str, Any]],
    scores: np.ndarray,
    y_pred: np.ndarray,
    threshold_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run, score, pred, threshold in zip(runs, scores, y_pred, threshold_values):
        rows.append(
            {
                "run_id": run["run_id"],
                "seed": run["seed"],
                "model": run["model"],
                "scenario": run["scenario"],
                "condition": run["condition"],
                "memory_mode": run["memory_mode"],
                "attack_family": run["attack_family"],
                "y_true": run["label"],
                "flagged": int(pred),
                "score": float(score),
                "threshold": float(threshold),
                "events_path": run["path"],
            }
        )
    return rows


def write_outputs(output_dir: Path, supervised_eval: dict[str, Any]) -> None:
    (output_dir / "supervised_evaluation.json").write_text(
        json.dumps(supervised_eval, indent=2), encoding="utf-8"
    )
    write_predictions(output_dir / "supervised_predictions.csv", supervised_eval["predictions"])
    write_detection_metrics(output_dir / "detection_metrics.csv", [supervised_eval])


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_detection_metrics(path: Path, evaluations: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row = {"model": evaluation["model"], **evaluation["test"]}
        rows.append(row)
        for family, values in evaluation["by_attack_family"].items():
            rows.append(
                {
                    "model": evaluation["model"],
                    "subset": f"attack_family={family}",
                    **values,
                }
            )
    fieldnames = sorted({key for row in rows for key in row})
    for row in rows:
        if isinstance(row.get("thresholds"), dict):
            row["thresholds"] = json.dumps(row["thresholds"], sort_keys=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_feature_table(path: Path, runs: list[dict[str, Any]]) -> None:
    keys = sorted({key for run in runs for key in run["features"]})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "seed", "model", "scenario", "condition", "memory_mode", "label", *keys],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_id": run["run_id"],
                    "seed": run["seed"],
                    "model": run["model"],
                    "scenario": run["scenario"],
                    "condition": run["condition"],
                    "memory_mode": run["memory_mode"],
                    "label": run["label"],
                    **run["features"],
                }
            )


def write_top_features(path: Path, model: Any, artifacts: dict[str, Any], limit: int = 80) -> None:
    names = feature_names(artifacts)
    rows: list[dict[str, Any]] = []
    if hasattr(model, "coef_"):
        weights = np.asarray(model.coef_[0], dtype=np.float64)
        order = np.argsort(np.abs(weights))[::-1][:limit]
        for index in order:
            rows.append(
                {
                    "feature": names[index],
                    "weight": float(weights[index]),
                    "direction": "attack" if weights[index] > 0 else "clean",
                    "absolute_weight": float(abs(weights[index])),
                }
            )
    elif hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_, dtype=np.float64)
        order = np.argsort(weights)[::-1][:limit]
        for index in order:
            rows.append(
                {
                    "feature": names[index],
                    "weight": float(weights[index]),
                    "direction": "importance",
                    "absolute_weight": float(weights[index]),
                }
            )
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_plots(output_dir: Path, supervised_eval: dict[str, Any]) -> None:
    write_metric_comparison(output_dir, [supervised_eval])
    write_roc_pr(output_dir, supervised_eval, "supervised")
    write_confusion(output_dir, supervised_eval, "supervised")
    write_family_bars(output_dir, supervised_eval, "supervised")


def write_metric_comparison(output_dir: Path, evaluations: list[dict[str, Any]]) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = [evaluation["model"].replace("_", " ") for evaluation in evaluations]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for offset, evaluation in enumerate(evaluations):
        values = [evaluation["test"].get(metric) or 0.0 for metric in metrics]
        ax.bar(x + (offset - 0.5) * width, values, width=width, label=labels[offset])
    ax.set_xticks(x, [metric.upper().replace("_", "@") for metric in metrics], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Agent-Log Detector Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "agent-log-detector-metrics")


def write_roc_pr(output_dir: Path, evaluation: dict[str, Any], prefix: str) -> None:
    rows = evaluation["predictions"]
    y_true = np.asarray([int(row["y_true"]) for row in rows])
    scores = np.asarray([float(row["score"]) for row in rows])
    if len(set(y_true.tolist())) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, color="#2f6fbb", linewidth=2.2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{prefix.title()} ROC")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / f"{prefix}-roc")

    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(recall, precision, color="#4f9d69", linewidth=2.2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{prefix.title()} Precision-Recall")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / f"{prefix}-pr")


def write_confusion(output_dir: Path, evaluation: dict[str, Any], prefix: str) -> None:
    rows = evaluation["predictions"]
    y_true = np.asarray([int(row["y_true"]) for row in rows])
    y_pred = np.asarray([int(row["flagged"]) for row in rows])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Clean", "Pred Attack"])
    ax.set_yticks([0, 1], labels=["True Clean", "True Attack"])
    ax.set_title(f"{prefix.title()} Confusion Matrix")
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() * 0.55 else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_dir / f"{prefix}-confusion-matrix")


def write_family_bars(output_dir: Path, evaluation: dict[str, Any], prefix: str) -> None:
    groups = evaluation["by_attack_family"]
    labels = list(groups)
    rates = [float(groups[label]["flagged_rate"]) for label in labels]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(labels, rates, color=["#777777" if label == "clean" else "#2f6fbb" for label in labels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Flagged Rate")
    ax.set_title("Flagged Rate By Attack Family")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
    save_figure(fig, output_dir / f"{prefix}-flagged-rate-by-family")


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=180)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close(fig)


def print_dataset_summary(
    runs: list[dict[str, Any]],
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    print(f"Loaded {len(runs)} runs")
    for name, split in (("train", train), ("validation", val), ("test", test)):
        y = labels(split)
        families = Counter(run["attack_family"] for run in split)
        print(
            f"{name}: runs={len(split)}, clean={int((y == 0).sum())}, "
            f"attack={int((y == 1).sum())}, seeds={sorted({run['seed'] for run in split})}, "
            f"families={dict(families)}"
        )


def print_result(evaluation: dict[str, Any]) -> None:
    test = evaluation["test"]
    print(f"\n{evaluation['model']}")
    if test.get("thresholds"):
        print(f"Thresholds: {json.dumps(test['thresholds'], sort_keys=True)}")
    else:
        print(f"Threshold: {test['threshold']:.6f}")
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test precision: {test['precision']:.4f}")
    print(f"Test recall: {test['recall']:.4f}")
    print(f"Test F1: {test['f1']:.4f}")
    print(f"Test ROC-AUC: {test['roc_auc']:.4f}" if test["roc_auc"] is not None else "Test ROC-AUC: n/a")
    print(f"Test PR-AUC: {test['pr_auc']:.4f}" if test["pr_auc"] is not None else "Test PR-AUC: n/a")
    print(
        "FPR@TPR=0.9: "
        + (f"{test['fpr_at_tpr_0_90']:.4f}" if test["fpr_at_tpr_0_90"] is not None else "n/a")
    )
    for family, values in evaluation["by_attack_family"].items():
        print(
            f"  {family}: n={values['runs']}, flagged={values['flagged_rate']:.2%}, "
            f"mean_score={values['mean_score']:.4f}"
        )


if __name__ == "__main__":
    main()
