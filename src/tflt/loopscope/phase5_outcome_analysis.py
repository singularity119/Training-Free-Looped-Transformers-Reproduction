"""One-shot LoopScope Phase 5 Gate D outcome analysis and verifier.

The module keeps preflight outcome-blind, binds the exact Gate C sealed panel,
and persists only aggregate paired statistics.  The ``execute`` entry point is
write-once: after its unseal marker exists, no second analysis is permitted.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import itertools
import json
import math
import random
import struct
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tflt.loopscope.analysis import AnalysisError, extract_accuracy, extract_correctness_samples
from tflt.loopscope.phase3_p3c import validate_test_metadata_records
from tflt.loopscope.phase3_schema import (
    canonical_json_bytes,
    file_sha256,
    ordered_identity_sha256,
)
from tflt.loopscope.phase5_outcome import CELLS, project_identity_only
from tflt.loopscope.phase5_schema import trajectory_window_signals


GATE = "D"
EXECUTOR_THREAD_ID = "019f8978-eb01-7ab3-b37e-9b797a40a262"
PLANNING_THREAD_ID = "019f8604-4717-7be2-8bf8-9d4a26a3d7f7"
AUTHORIZED_BASE_COMMIT = "684bd0ccaf1e508a645d0ab4d1884ea5939fc9c0"

REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
WORKSPACE = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope"
)
GATE_C_ROOT = WORKSPACE / "runs/phase5-gate-c-repair-identity-20260722T072256Z"
GATE_D_ROOT = GATE_C_ROOT / "gate_d"
GATE_B_ROOT = WORKSPACE / "runs/phase5-gate-b-repair-b36-native-20260722T035452Z"
PHASE3_C_ROOT = WORKSPACE / "runs/phase3-p3c-20260716T064601Z"

CARD_PATH = Path("configs/loopscope/phase5_card.json")
CARD_FILE_SHA256 = "8d25996f994d131ea0fd9ca78f83c90680a67c41c7c460f7e4e8b4ae5ea42a73"
CARD_MANIFEST_SHA256 = "75eed3e30c623f5eedd63e84e93bfd3e4de50cfb490b23f3a2d31683f9947143"
SELECTOR_REPORT_PATH = GATE_B_ROOT / "selector/selector_report.json"
SELECTOR_REPORT_FILE_SHA256 = "a6e7eb396b1da90152c2dc56a73e1866b3e98a40f044cfcc29e28091ae16beb3"
SELECTOR_REPORT_MANIFEST_SHA256 = "1505319c9c300e146e7158327c726c82e9af4f477f1bd5163ad78bbe6f4b43b0"
SELECTOR_FREEZE_PATH = GATE_B_ROOT / "selector/selector_freeze.json"
SELECTOR_FREEZE_FILE_SHA256 = "7c42f0e22ce2caa5784edb33dcfc271e31291229e107b478928fd77181cf1a0b"
SELECTOR_FREEZE_MANIFEST_SHA256 = "4c66ef75ee8efc492245b341ee79c0147a3ba3b973a0de569ef6ca5dcb858d97"
PANEL_PATH = GATE_B_ROOT / "selector/outcome_panel.json"
PANEL_FILE_SHA256 = "3ea83d12e5185edffbf01d5b916612e616d3a7cf41881601fa7b8931918be0e9"
PANEL_MANIFEST_SHA256 = "f28ce84d54e69f2179209f276ecd69f39c3e94359eac4f2aed84b09167be4961"
TRAJECTORY_PATH = GATE_B_ROOT / "formal/validation1531_phase5_trajectories.jsonl"
TRAJECTORY_FILE_SHA256 = "aa4894e632f557ddc4614e88d06ec1fc4c525463be7531aef69fa3c28671edf9"
TEST_MANIFEST_PATH = PHASE3_C_ROOT / "test14042_identity_content_manifest.json"
TEST_MANIFEST_FILE_SHA256 = "59500340aa64d91810cd5a5d40f3d96ded5618ab5508543b31eea93050e7f3e4"
TEST_MANIFEST_INTERNAL_SHA256 = "ec494ff6dd55b4398cd8151eed4609916e8d4c272df4583a4a1574cab3c25386"
TEST_METADATA_PATH = PHASE3_C_ROOT / "test14042_identity_content_metadata.jsonl"
TEST_METADATA_FILE_SHA256 = "3c281447cc7618e58c292a20c13b73d8f42f081692a6927372cf1ee1eebf84df"
TEST_ORDERED_IDENTITY_SHA256 = "2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532"
COMPLETION_PATH = GATE_C_ROOT / "phase5_gate_c_completion_receipt.json"
COMPLETION_FILE_SHA256 = "4f15ae291afb2fcb00992cfd3bb4d216386874cfb1ff55bf1d3a9ad7c8e9c8d4"
COMPLETION_MANIFEST_SHA256 = "979b4e61fc32b3eeb2cebdd5d6a5662f9bdbc0c87eb296a67f7be97ad0ee0ac6"

EXPECTED_COUNT = 14042
EXPECTED_SUBJECTS = 57
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260723
TERMINAL_LABEL = "H5_ABSTAIN_WITH_RANKING_RESULT"
BLIND_WINDOWS = ("4:7", "5:8", "13:16", "8:11", "9:12", "6:9")

UNSEAL_NAME = "phase5_gate_d_unseal_once.json"
ANALYSIS_NAME = "phase5_gate_d_analysis.json"
TABLE_NAME = "phase5_gate_d_window_metrics.csv"
BOOTSTRAP_NAME = "phase5_gate_d_bootstrap_digest.json"
VERIFIER_NAME = "phase5_gate_d_verifier_receipt.json"
SUMMARY_NAME = "phase5_gate_d_summary_zh.md"
TERMINAL_NAME = "phase5_gate_d_terminal_receipt.json"

IMPLEMENTATION_PATHS = (
    "src/tflt/loopscope/phase5_outcome_analysis.py",
    "scripts/loopscope/run_qwen4base_phase5_gate_d.py",
    "tests/test_loopscope_phase5_gate_d.py",
)


class Phase5GateDError(ValueError):
    """Fail-closed Gate D contract, identity, or analysis error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise Phase5GateDError("git command failed: %s" % " ".join(args))
    return completed.stdout.strip()


def git_provenance(expected_commit: str, *, remote_required: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if root != REMOTE_REPO.resolve() and str(root) != str(REMOTE_REPO):
        # Local implementation/test runs use the dedicated Mac clone.
        expected_local = Path(
            "/Users/huangxutao/Desktop/Training-free looped transformer/"
            "LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/"
            "loopscope-tflt"
        )
        if root != expected_local:
            raise Phase5GateDError("repository root differs from the dedicated LoopScope clone")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--short"))
    origin = _git(root, "rev-parse", "origin/loopscope") if remote_required else None
    if branch != "loopscope" or commit != expected_commit or dirty:
        raise Phase5GateDError("Gate D requires clean loopscope at expected commit")
    if remote_required and origin != expected_commit:
        raise Phase5GateDError("origin/loopscope differs from expected commit")
    return {
        "repo": str(root), "branch": branch, "commit": commit,
        "dirty": dirty, "origin_loopscope": origin,
    }


def _strict_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError, TypeError) as exc:
        raise Phase5GateDError("invalid JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise Phase5GateDError("JSON root must be an object: %s" % path)
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line, parse_constant=_reject_constant)
            except (ValueError, TypeError) as exc:
                raise Phase5GateDError("invalid JSONL row %d: %s" % (number, path)) from exc
            if not isinstance(value, dict):
                raise Phase5GateDError("JSONL row must be an object")
            rows.append(value)
    return rows


def _manifest_sha256(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
        raise Phase5GateDError("%s file SHA256 differs" % label)


def _require_manifest(value: Mapping[str, Any], expected: str, label: str) -> None:
    if value.get("manifest_sha256") != expected or _manifest_sha256(value) != expected:
        raise Phase5GateDError("%s internal manifest SHA256 differs" % label)


def implementation_hashes() -> Dict[str, str]:
    result = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository_root() / relative
        if not path.is_file():
            raise Phase5GateDError("missing Gate D implementation path: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def _load_test_metadata(card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = _jsonl(TEST_METADATA_PATH)
    try:
        normalized = validate_test_metadata_records(
            rows,
            card,
            enforce_frozen_counts=True,
            expected_ordered_identity_sha256=TEST_ORDERED_IDENTITY_SHA256,
        )
    except Exception as exc:
        raise Phase5GateDError("test metadata contract differs") from exc
    return normalized


def validate_inputs(*, hash_results: bool = True) -> Dict[str, Any]:
    """Validate every pre-unseal binding without parsing a results payload."""

    root = repository_root()
    bindings = (
        (root / CARD_PATH, CARD_FILE_SHA256, "card"),
        (SELECTOR_REPORT_PATH, SELECTOR_REPORT_FILE_SHA256, "selector report"),
        (SELECTOR_FREEZE_PATH, SELECTOR_FREEZE_FILE_SHA256, "selector freeze"),
        (PANEL_PATH, PANEL_FILE_SHA256, "outcome panel"),
        (TRAJECTORY_PATH, TRAJECTORY_FILE_SHA256, "trajectory"),
        (TEST_MANIFEST_PATH, TEST_MANIFEST_FILE_SHA256, "test manifest"),
        (TEST_METADATA_PATH, TEST_METADATA_FILE_SHA256, "test metadata"),
        (COMPLETION_PATH, COMPLETION_FILE_SHA256, "completion receipt"),
    )
    for path, expected, label in bindings:
        _require_hash(path, expected, label)

    card = _strict_json(root / CARD_PATH)
    report = _strict_json(SELECTOR_REPORT_PATH)
    freeze = _strict_json(SELECTOR_FREEZE_PATH)
    panel = _strict_json(PANEL_PATH)
    test_manifest = _strict_json(TEST_MANIFEST_PATH)
    completion = _strict_json(COMPLETION_PATH)
    _require_manifest(card, CARD_MANIFEST_SHA256, "card")
    _require_manifest(report, SELECTOR_REPORT_MANIFEST_SHA256, "selector report")
    _require_manifest(freeze, SELECTOR_FREEZE_MANIFEST_SHA256, "selector freeze")
    _require_manifest(panel, PANEL_MANIFEST_SHA256, "outcome panel")
    _require_manifest(test_manifest, TEST_MANIFEST_INTERNAL_SHA256, "test manifest")
    _require_manifest(completion, COMPLETION_MANIFEST_SHA256, "completion receipt")

    expected_cells = [
        {"cell_id": cell_id, "role": role, "window": window}
        for cell_id, role, window in CELLS
    ]
    observed_cells = [
        {"cell_id": row.get("cell_id"), "role": row.get("role"), "window": row.get("window")}
        for row in completion.get("cells", [])
    ]
    if observed_cells != expected_cells:
        raise Phase5GateDError("Gate C completion cell membership/order differs")
    identity = completion.get("identity_closure", {})
    if (
        completion.get("status") != "SEALED_COMPLETE_READY_FOR_PLANNING_AUDIT"
        or completion.get("cell_count") != 8
        or completion.get("outcome_values_consumed") is not False
        or completion.get("accuracy_or_gain_computed") is not False
        or completion.get("results_payload_parsed_by_sealer") is not False
        or completion.get("gate_d_unseal_performed") is not False
        or identity.get("record_count_per_cell") != EXPECTED_COUNT
        or identity.get("subject_count_per_cell") != EXPECTED_SUBJECTS
        or identity.get("missing_duplicate_extra") != 0
        or identity.get("ordered_identity_sha256_all_cells") != TEST_ORDERED_IDENTITY_SHA256
    ):
        raise Phase5GateDError("Gate C sealed completion closure differs")
    if (
        freeze.get("window_decision") != "ABSTAIN_NO_POINT_ELIGIBLE"
        or freeze.get("selected_window") is not None
        or freeze.get("eligible_count") != 0
        or freeze.get("selection_frequency") is not None
        or panel.get("unique_cells")
        != ["baseline", "15:18", "4:7", "5:8", "13:16", "8:11", "9:12", "6:9"]
        or panel.get("panel_high3") != ["4:7", "5:8", "13:16"]
        or panel.get("blind_low3") != ["8:11", "9:12", "6:9"]
    ):
        raise Phase5GateDError("frozen ABSTAIN or panel contract differs")
    if (
        test_manifest.get("record_count") != EXPECTED_COUNT
        or test_manifest.get("subject_count") != EXPECTED_SUBJECTS
        or test_manifest.get("ordered_identity_sha256") != TEST_ORDERED_IDENTITY_SHA256
    ):
        raise Phase5GateDError("test manifest population closure differs")

    result_files: Dict[str, str] = {}
    for row in completion["cells"]:
        result = row.get("results", {})
        path = Path(str(result.get("path", "")))
        expected_path = GATE_C_ROOT / "formal" / (
            "cell-%02d-%s" % (int(row["index"]), str(row["cell_id"]))
        ) / "eval/results.json"
        if path != expected_path or not path.is_file() or path.is_symlink():
            raise Phase5GateDError("sealed result path differs for %s" % row["cell_id"])
        if path.stat().st_size != result.get("size_bytes"):
            raise Phase5GateDError("sealed result size differs for %s" % row["cell_id"])
        if hash_results and file_sha256(path) != result.get("file_sha256"):
            raise Phase5GateDError("sealed result SHA256 differs for %s" % row["cell_id"])
        result_files[str(row["cell_id"])] = str(result.get("file_sha256"))

    metadata = _load_test_metadata(card)
    return {
        "card": card,
        "selector_report": report,
        "selector_freeze": freeze,
        "panel": panel,
        "completion": completion,
        "test_metadata": metadata,
        "result_file_sha256": result_files,
        "source_file_sha256": {label: expected for _, expected, label in bindings},
    }


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    numbers = sorted(float(value) for value in values)
    if not numbers or not 0.0 <= quantile <= 1.0:
        raise Phase5GateDError("invalid percentile input")
    position = quantile * (len(numbers) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return numbers[lower]
    weight = position - lower
    return numbers[lower] * (1.0 - weight) + numbers[upper] * weight


def exact_mcnemar_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    values = (wrong_to_correct, correct_to_wrong)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise Phase5GateDError("McNemar counts must be non-negative integers")
    total = sum(values)
    if not total:
        return 1.0
    tail = min(values)
    logs = [
        math.lgamma(total + 1) - math.lgamma(index + 1)
        - math.lgamma(total - index + 1) - total * math.log(2.0)
        for index in range(tail + 1)
    ]
    peak = max(logs)
    one_sided = math.exp(peak) * math.fsum(math.exp(value - peak) for value in logs)
    return min(1.0, 2.0 * one_sided)


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        rank = 0.5 * (offset + 1 + end)
        for index in order[offset:end]:
            ranks[index] = rank
        offset = end
    return ranks


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise Phase5GateDError("Spearman vectors must have equal length >=2")
    x, y = _average_ranks(left), _average_ranks(right)
    x_mean, y_mean = math.fsum(x) / len(x), math.fsum(y) / len(y)
    numerator = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_ss = math.fsum((value - x_mean) ** 2 for value in x)
    y_ss = math.fsum((value - y_mean) ** 2 for value in y)
    if x_ss == 0.0 or y_ss == 0.0:
        raise Phase5GateDError("Spearman vector is constant")
    return numerator / math.sqrt(x_ss * y_ss)


def exact_spearman(scores: Sequence[float], gains: Sequence[float]) -> Dict[str, Any]:
    if len(scores) != 6 or len(gains) != 6:
        raise Phase5GateDError("blind Spearman requires exactly six windows")
    observed = spearman_rho(scores, gains)
    extreme = 0
    count = 0
    for order in itertools.permutations(range(6)):
        rho = spearman_rho(scores, [gains[index] for index in order])
        count += 1
        if abs(rho) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "method": "exhaustive_exact_two_sided_spearman_permutation",
        "rho": observed, "permutation_count": count,
        "extreme_count_abs_rho_ge_observed": extreme, "p_value": extreme / count,
    }


def _bootstrap_summary(values: Sequence[float]) -> Dict[str, Any]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {
        "bootstrap_mean_fraction": mean,
        "bootstrap_standard_error_ddof1": math.sqrt(variance),
        "percentile_95_ci_fraction": [
            linear_percentile(values, 0.025), linear_percentile(values, 0.975)
        ],
    }


def subject_stratified_joint_bootstrap(
    correctness: np.ndarray,
    subjects: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    matrix = np.asarray(correctness, dtype=np.int8)
    if matrix.shape != (len(subjects), 8) or len(subjects) < 2:
        raise Phase5GateDError("bootstrap matrix shape differs")
    if not np.logical_or(matrix == 0, matrix == 1).all():
        raise Phase5GateDError("bootstrap matrix is not binary")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise Phase5GateDError("bootstrap replicates must be >=2")
    groups = {
        subject: [index for index, value in enumerate(subjects) if value == subject]
        for subject in sorted(set(subjects))
    }
    rng = random.Random(seed)
    digest = hashlib.sha256()
    estimates = np.empty((replicates, 8), dtype=np.float64)
    for replicate in range(replicates):
        drawn: List[int] = []
        for subject in sorted(groups):
            indices = groups[subject]
            drawn.extend(indices[rng.randrange(len(indices))] for _ in indices)
        digest.update(struct.pack("<%dI" % len(drawn), *drawn))
        estimates[replicate] = matrix[np.asarray(drawn, dtype=np.int64)].mean(axis=0)
    gains = estimates[:, 1:] - estimates[:, [0]]
    high = gains[:, 1:4].mean(axis=1)
    low = gains[:, 4:7].mean(axis=1)
    enrichment = high - low
    return {
        "method": "subject_stratified_joint_paired_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "prng": "python_stdlib_random.Random",
        "random_api": "randrange(n_s)",
        "subject_order": sorted(groups),
        "subject_sizes": {key: len(groups[key]) for key in sorted(groups)},
        "draw_iteration_order": "replicate_major_subject_ascending_identity_canonical",
        "draw_digest_representation": "little_endian_uint32_global_indices",
        "draw_index_sha256": digest.hexdigest(),
        "same_joint_draw_stream_all_cells_and_contrasts": True,
        "standard_error": "sample_sd_ddof_1",
        "percentile_method": "linear_interpolation_at_q_times_R_minus_1",
        "cell_gain_summaries": [_bootstrap_summary(gains[:, index].tolist()) for index in range(7)],
        "high3_mean_gain": _bootstrap_summary(high.tolist()),
        "low3_mean_gain": _bootstrap_summary(low.tolist()),
        "high3_minus_low3": _bootstrap_summary(enrichment.tolist()),
    }


def _transitions(baseline: np.ndarray, candidate: np.ndarray) -> Dict[str, int]:
    return {
        "wrong_to_correct": int(np.logical_and(baseline == 0, candidate == 1).sum()),
        "correct_to_wrong": int(np.logical_and(baseline == 1, candidate == 0).sum()),
        "correct_to_correct": int(np.logical_and(baseline == 1, candidate == 1).sum()),
        "wrong_to_wrong": int(np.logical_and(baseline == 0, candidate == 0).sum()),
    }


def _ranking_context(selector_report: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, int]]:
    ranking = selector_report.get("published_ranking")
    if not isinstance(ranking, list):
        raise Phase5GateDError("selector report lacks published ranking")
    scores, ranks = {}, {}
    for rank, row in enumerate(ranking, start=1):
        if not isinstance(row, Mapping):
            raise Phase5GateDError("selector ranking row is malformed")
        window = str(row.get("window"))
        score = float(row.get("score"))
        if not math.isfinite(score) or window in scores:
            raise Phase5GateDError("selector ranking score is malformed")
        scores[window], ranks[window] = score, rank
    if not set(BLIND_WINDOWS).issubset(scores):
        raise Phase5GateDError("selector ranking lacks a blind panel window")
    return scores, ranks


def _geometry_context() -> Dict[str, Dict[str, float]]:
    records = _jsonl(TRAJECTORY_PATH)
    accum = {
        window: {"E": [], "K": [], "hidden_l2_improvement": [],
                 "hidden_cosine_improvement": [], "hidden_cosine_distance_improvement": []}
        for window in BLIND_WINDOWS
    }
    for record in records:
        signals = trajectory_window_signals(record)["windows"]
        by_window = {row["window"]: row for row in signals}
        for window in BLIND_WINDOWS:
            row = by_window[window]
            accum[window]["E"].append(float(row["E"]))
            accum[window]["K"].append(float(row["K"]))
            accum[window]["hidden_l2_improvement"].append(-float(row["delta_hidden_l2_to_final"]))
            accum[window]["hidden_cosine_improvement"].append(float(row["delta_hidden_cosine_to_final"]))
            accum[window]["hidden_cosine_distance_improvement"].append(
                -float(row["delta_hidden_cosine_distance_to_final"])
            )
    if len(records) != 1531:
        raise Phase5GateDError("geometry trajectory count differs")
    return {
        window: {name: math.fsum(values) / len(values) for name, values in metrics.items()}
        for window, metrics in accum.items()
    }


def _load_correctness_cell(
    result_path: Path,
    metadata: Sequence[Mapping[str, Any]],
) -> List[bool]:
    payload = _strict_json(result_path)
    project_identity_only(payload, metadata)
    try:
        correctness = extract_correctness_samples(payload, task="mmlu", metric="acc,none")
        aggregate = extract_accuracy(payload, task="mmlu", metric="acc,none")
    except AnalysisError as exc:
        raise Phase5GateDError("lm-eval outcome payload cannot close acc,none") from exc
    ordered = []
    for row in metadata:
        identity = row["identity"]
        key = "%s:%s" % (identity["task"], identity["doc_id"])
        if key not in correctness:
            raise Phase5GateDError("outcome correctness lacks canonical identity")
        ordered.append(bool(correctness[key]))
    if set(correctness) != {
        "%s:%s" % (row["identity"]["task"], row["identity"]["doc_id"])
        for row in metadata
    }:
        raise Phase5GateDError("outcome correctness has missing or extra identity")
    sample_accuracy = math.fsum(ordered) / len(ordered)
    if not math.isclose(sample_accuracy, aggregate, rel_tol=0.0, abs_tol=1e-12):
        raise Phase5GateDError("aggregate and per-sample acc,none differ")
    del payload, correctness
    gc.collect()
    return ordered


def _correctness_matrix(inputs: Mapping[str, Any]) -> np.ndarray:
    columns = []
    for row in inputs["completion"]["cells"]:
        columns.append(
            _load_correctness_cell(Path(row["results"]["path"]), inputs["test_metadata"])
        )
    matrix = np.column_stack([np.asarray(column, dtype=np.int8) for column in columns])
    if matrix.shape != (EXPECTED_COUNT, 8):
        raise Phase5GateDError("outcome matrix does not close 14,042 x 8")
    return matrix


def build_statistical_analysis(
    *,
    correctness: np.ndarray,
    identities: Sequence[Mapping[str, Any]],
    subjects: Sequence[str],
    selector_report: Mapping[str, Any],
    geometry: Mapping[str, Mapping[str, float]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    matrix = np.asarray(correctness, dtype=np.int8)
    if matrix.shape != (len(identities), 8) or len(identities) != len(subjects):
        raise Phase5GateDError("analysis input shape differs")
    scores, ranks = _ranking_context(selector_report)
    bootstrap = subject_stratified_joint_bootstrap(
        matrix, subjects, replicates=replicates, seed=seed
    )
    accuracy = matrix.mean(axis=0, dtype=np.float64)
    gains = accuracy[1:] - accuracy[0]
    cells = []
    for index, (cell_id, role, window) in enumerate(CELLS):
        if index == 0:
            gain, ci, p_value = 0.0, [0.0, 0.0], 1.0
            transitions = {
                "wrong_to_correct": 0, "correct_to_wrong": 0,
                "correct_to_correct": int(matrix[:, 0].sum()),
                "wrong_to_wrong": int(len(matrix) - matrix[:, 0].sum()),
            }
        else:
            gain = float(gains[index - 1])
            ci = bootstrap["cell_gain_summaries"][index - 1]["percentile_95_ci_fraction"]
            transitions = _transitions(matrix[:, 0], matrix[:, index])
            p_value = exact_mcnemar_p(
                transitions["wrong_to_correct"], transitions["correct_to_wrong"]
            )
        cells.append({
            "index": index, "cell_id": cell_id, "role": role, "window": window,
            "sample_count": len(matrix), "accuracy_fraction": float(accuracy[index]),
            "accuracy_percent": float(accuracy[index] * 100.0),
            "gain_fraction": gain, "gain_percentage_points": gain * 100.0,
            "paired_percentile_95_ci_fraction": list(ci),
            "paired_percentile_95_ci_percentage_points": [value * 100.0 for value in ci],
            "transitions_vs_baseline": transitions,
            "exact_two_sided_mcnemar_p": p_value,
            "selector_score": None if window is None else scores[window],
            "selector_rank": None if window is None else ranks[window],
        })

    blind_rows = [row for row in cells if row["role"] in {"blind_high", "blind_low"}]
    blind_scores = [float(row["selector_score"]) for row in blind_rows]
    blind_gains = [float(row["gain_fraction"]) for row in blind_rows]
    rho = exact_spearman(blind_scores, blind_gains)
    high_point = float(gains[1:4].mean())
    low_point = float(gains[4:7].mean())
    enrich_point = high_point - low_point
    enrich_ci = bootstrap["high3_minus_low3"]["percentile_95_ci_fraction"]
    best_accuracy = max(row["accuracy_fraction"] for row in cells[1:])
    best = [row for row in cells[1:] if row["accuracy_fraction"] == best_accuracy]
    raw_top = next(row for row in cells if row["window"] == "4:7")
    geometry_rows = [geometry[window] for window in BLIND_WINDOWS]
    geometry_diagnostics: Dict[str, Any] = {
        "role": "secondary_diagnostic_not_selector_or_terminal_success",
        "windows": list(BLIND_WINDOWS),
        "per_window_validation_means": {window: dict(geometry[window]) for window in BLIND_WINDOWS},
        "correlations": {},
    }
    for name in (
        "hidden_l2_improvement", "hidden_cosine_improvement",
        "hidden_cosine_distance_improvement",
    ):
        values = [float(row[name]) for row in geometry_rows]
        geometry_diagnostics["correlations"][name] = {
            "vs_output_E_spearman": spearman_rho(values, [float(row["E"]) for row in geometry_rows]),
            "vs_output_K_spearman": spearman_rho(values, [float(row["K"]) for row in geometry_rows]),
            "vs_test_gain_spearman": spearman_rho(values, blind_gains),
        }
    return {
        "schema_version": "loopscope.phase5.gate-d-analysis.v1",
        "artifact_role": "one_shot_frozen_eight_cell_subject_stratified_paired_analysis",
        "gate": GATE,
        "population": {
            "record_count": len(matrix), "subject_count": len(set(subjects)),
            "cell_count": 8, "matrix_shape": [len(matrix), 8],
            "ordered_identity_sha256": ordered_identity_sha256(identities),
            "subject_counts": dict(sorted(Counter(subjects).items())),
            "missing_duplicate_extra": 0,
        },
        "metric": {"primary": "acc,none", "internal_unit": "fraction", "report_unit": "percentage_points"},
        "cells": cells,
        "groups": {
            "high3": {
                "windows": ["4:7", "5:8", "13:16"],
                "mean_gain_fraction": high_point,
                "mean_gain_percentage_points": high_point * 100.0,
                "paired_percentile_95_ci_fraction": bootstrap["high3_mean_gain"]["percentile_95_ci_fraction"],
            },
            "low3": {
                "windows": ["8:11", "9:12", "6:9"],
                "mean_gain_fraction": low_point,
                "mean_gain_percentage_points": low_point * 100.0,
                "paired_percentile_95_ci_fraction": bootstrap["low3_mean_gain"]["percentile_95_ci_fraction"],
            },
            "high3_minus_low3": {
                "point_fraction": enrich_point,
                "point_percentage_points": enrich_point * 100.0,
                "paired_percentile_95_ci_fraction": list(enrich_ci),
                "paired_percentile_95_ci_percentage_points": [value * 100.0 for value in enrich_ci],
                "positive_point": enrich_point > 0.0,
                "ci_excludes_zero_positive": enrich_ci[0] > 0.0,
                "ci_crosses_or_touches_zero": enrich_ci[0] <= 0.0 <= enrich_ci[1],
            },
        },
        "bootstrap": bootstrap,
        "blind_six_selector_score_vs_gain": {**rho, "windows": [row["window"] for row in blind_rows], "fixed_15_18_excluded": True},
        "abstain": {
            "selected_window": None,
            "window_decision": "ABSTAIN_NO_POINT_ELIGIBLE",
            "eligible_count": 0,
            "selection_frequency": None,
            "coverage": 0.0,
            "selected_vs_baseline": None,
            "selected_vs_15_18": None,
            "selected_panel_local_regret": None,
            "not_applicable_reason": "legal_frozen_abstain_no_selected_window",
        },
        "ranking_diagnostics": {
            "raw_high3_top1_window": "4:7",
            "raw_high3_top1_gain_percentage_points": raw_top["gain_percentage_points"],
            "panel_best_window": best[0]["window"],
            "panel_best_windows": [row["window"] for row in best],
            "panel_best_tied": len(best) > 1,
            "raw_top1_panel_local_regret_percentage_points": (best_accuracy - raw_top["accuracy_fraction"]) * 100.0,
            "not_selection_success": True,
        },
        "known_comparator": {
            "window": "15:18",
            "current_accuracy_percent": cells[1]["accuracy_percent"],
            "current_gain_percentage_points": cells[1]["gain_percentage_points"],
            "public_73_61_and_plus_0_34_pp_role": "background_only_not_acceptance_threshold",
        },
        "absolute_gain": {
            "high3_mean_gain_positive": high_point > 0.0,
            "high3_mean_gain_percentage_points": high_point * 100.0,
            "high3_mean_gain_ci_percentage_points": [
                value * 100.0 for value in bootstrap["high3_mean_gain"]["percentile_95_ci_fraction"]
            ],
        },
        "geometry_diagnostics": geometry_diagnostics,
        "terminal": {
            "label": TERMINAL_LABEL,
            "prospective_selection_supported_forbidden_by_abstain": True,
            "ranking_enrichment_reported_separately": True,
            "absolute_gain_reported_separately": True,
            "known_comparator_reported_separately": True,
        },
        "claim_boundary": {
            "cell": "Qwen3-4B-Base_x_MMLU_5shot",
            "scope": "within_cell_prospective_and_limited_cross_scale_consistency",
            "universal_cross_model_automatic_window_selection_claimed": False,
        },
        "persistence_safety": {
            "raw_payload_copied": False, "row_level_correctness_persisted": False,
            "full_logits_or_hidden_states_persisted": False, "selector_or_panel_changed": False,
        },
    }


def _write_new_text(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
    return file_sha256(path)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _write_new_text(path, text)


def _csv_text(cells: Sequence[Mapping[str, Any]]) -> str:
    from io import StringIO
    buffer = StringIO(newline="")
    fields = (
        "cell_id", "role", "window", "accuracy_percent", "gain_percentage_points",
        "ci95_low_pp", "ci95_high_pp", "wrong_to_correct", "correct_to_wrong",
        "correct_to_correct", "wrong_to_wrong", "exact_two_sided_mcnemar_p",
        "selector_score", "selector_rank",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in cells:
        transitions = row["transitions_vs_baseline"]
        ci = row["paired_percentile_95_ci_percentage_points"]
        writer.writerow({
            "cell_id": row["cell_id"], "role": row["role"], "window": row["window"],
            "accuracy_percent": row["accuracy_percent"],
            "gain_percentage_points": row["gain_percentage_points"],
            "ci95_low_pp": ci[0], "ci95_high_pp": ci[1],
            **transitions,
            "exact_two_sided_mcnemar_p": row["exact_two_sided_mcnemar_p"],
            "selector_score": row["selector_score"], "selector_rank": row["selector_rank"],
        })
    return buffer.getvalue()


def _summary_zh(analysis: Mapping[str, Any]) -> str:
    group = analysis["groups"]["high3_minus_low3"]
    high = analysis["groups"]["high3"]
    rho = analysis["blind_six_selector_score_vs_gain"]
    lines = [
        "# LoopScope 第五阶段 Gate D 一次性结果摘要",
        "",
        "终端标签：`%s`。冻结 selector 合法 `ABSTAIN_NO_POINT_ELIGIBLE`，因此本阶段不把 raw rank-1 或 panel best 冒充 selected success。" % TERMINAL_LABEL,
        "",
        "- High3 平均相对 baseline 收益：`%.6f pp`。" % high["mean_gain_percentage_points"],
        "- High3−Low3 enrichment：`%.6f pp`，95%% CI `[%.6f, %.6f] pp`。" % (
            group["point_percentage_points"], *group["paired_percentile_95_ci_percentage_points"]
        ),
        "- blind-six selector score–gain Spearman：`rho=%.6f`，exact two-sided `p=%.6f`。" % (rho["rho"], rho["p_value"]),
        "- `selected_vs_baseline`、`selected_vs_15_18`、selected panel-local regret 均为 `not applicable`；这是 ABSTAIN 的语义，不是缺失结果。",
        "- 公开的 `73.27 / 73.61 / +0.34 pp` 只作背景，当前八-cell 逐样本结果独立报告。",
        "- geometry/output 与 geometry/gain 相关性仅为 secondary diagnostic，不进入 selector 或 terminal success 判据。",
        "",
        "结论边界：证据只覆盖 `Qwen3-4B-Base × MMLU 5-shot` 当前 cell 的 prospective / limited cross-scale consistency；不声称获得跨模型通用自动选窗器。",
        "",
        "完整 accuracy、paired gain、CI、flip 与 McNemar 表见 `phase5_gate_d_window_metrics.csv`。",
    ]
    return "\n".join(lines) + "\n"


def _scientific_projection(analysis: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in analysis.items()
        if key not in {"created_at_utc", "argv", "bindings", "input_paths"}
    }


def run_one_shot(*, expected_commit: str, argv: Sequence[str]) -> Dict[str, Any]:
    if GATE_D_ROOT.exists():
        raise FileExistsError("Gate D output root already exists; refusing second unseal")
    git = git_provenance(expected_commit, remote_required=True)
    inputs = validate_inputs(hash_results=True)
    GATE_D_ROOT.mkdir(parents=False, exist_ok=False)
    marker = {
        "schema_version": "loopscope.phase5.gate-d-unseal-once.v1",
        "gate": GATE, "executor_thread_id": EXECUTOR_THREAD_ID,
        "created_at_utc": utc_now(), "git": git, "argv": list(argv),
        "source_file_sha256": inputs["source_file_sha256"],
        "sealed_result_file_sha256": inputs["result_file_sha256"],
        "completion_receipt_file_sha256": COMPLETION_FILE_SHA256,
        "completion_receipt_manifest_sha256": COMPLETION_MANIFEST_SHA256,
        "implementation_sha256": implementation_hashes(),
        "unseal_count": 1, "analysis_count": 1,
        "status": "ONE_SHOT_UNSEAL_STARTED_NO_RERUN_PERMITTED",
    }
    marker_sha = _write_new_json(GATE_D_ROOT / UNSEAL_NAME, marker)

    matrix = _correctness_matrix(inputs)
    metadata = inputs["test_metadata"]
    identities = [row["identity"] for row in metadata]
    subjects = [str(row["subject"]) for row in metadata]
    geometry = _geometry_context()
    analysis = build_statistical_analysis(
        correctness=matrix, identities=identities, subjects=subjects,
        selector_report=inputs["selector_report"], geometry=geometry,
    )
    analysis.update({
        "created_at_utc": utc_now(), "argv": list(argv),
        "bindings": {
            "implementation_commit": expected_commit,
            "implementation_sha256": implementation_hashes(),
            "card_file_sha256": CARD_FILE_SHA256,
            "selector_report_file_sha256": SELECTOR_REPORT_FILE_SHA256,
            "selector_freeze_file_sha256": SELECTOR_FREEZE_FILE_SHA256,
            "panel_file_sha256": PANEL_FILE_SHA256,
            "completion_receipt_file_sha256": COMPLETION_FILE_SHA256,
            "completion_receipt_manifest_sha256": COMPLETION_MANIFEST_SHA256,
        },
        "input_paths": {
            "completion_receipt": str(COMPLETION_PATH),
            "selector_report": str(SELECTOR_REPORT_PATH),
            "selector_freeze": str(SELECTOR_FREEZE_PATH),
            "outcome_panel": str(PANEL_PATH),
            "test_manifest": str(TEST_MANIFEST_PATH),
        },
    })
    analysis_sha = _write_new_json(GATE_D_ROOT / ANALYSIS_NAME, analysis)
    table_sha = _write_new_text(GATE_D_ROOT / TABLE_NAME, _csv_text(analysis["cells"]))
    bootstrap_artifact = {
        "schema_version": "loopscope.phase5.gate-d-bootstrap-digest.v1",
        **analysis["bootstrap"],
    }
    bootstrap_sha = _write_new_json(GATE_D_ROOT / BOOTSTRAP_NAME, bootstrap_artifact)
    summary_sha = _write_new_text(GATE_D_ROOT / SUMMARY_NAME, _summary_zh(analysis))
    del matrix
    gc.collect()

    # Independent pass: reread all raw sealed payloads and reconstruct every statistic.
    verify_inputs = validate_inputs(hash_results=True)
    verify_matrix = _correctness_matrix(verify_inputs)
    verify_geometry = _geometry_context()
    recomputed = build_statistical_analysis(
        correctness=verify_matrix,
        identities=[row["identity"] for row in verify_inputs["test_metadata"]],
        subjects=[str(row["subject"]) for row in verify_inputs["test_metadata"]],
        selector_report=verify_inputs["selector_report"], geometry=verify_geometry,
    )
    if canonical_json_bytes(_scientific_projection(analysis)) != canonical_json_bytes(recomputed):
        raise Phase5GateDError("independent verifier reconstruction differs from analysis")
    if _csv_text(recomputed["cells"]) != (GATE_D_ROOT / TABLE_NAME).read_text(encoding="utf-8"):
        raise Phase5GateDError("independent verifier CSV reconstruction differs")
    if recomputed["bootstrap"]["draw_index_sha256"] != bootstrap_artifact["draw_index_sha256"]:
        raise Phase5GateDError("independent verifier bootstrap stream differs")
    verifier = {
        "schema_version": "loopscope.phase5.gate-d-verifier-receipt.v1",
        "gate": GATE, "created_at_utc": utc_now(), "git": git,
        "verifier_invocation_count": 1, "raw_payload_reconstruction_count": 1,
        "checks": {
            "exact_14042_x_8_identity_equality": True,
            "cell_accuracy_gain_flip_mcnemar_recomputed": True,
            "bootstrap_draw_digest_and_contrasts_recomputed": True,
            "blind_score_gain_spearman_recomputed": True,
            "abstain_fields_and_terminal_label_recomputed": True,
            "claim_boundary_flags_recomputed": True,
            "no_row_level_outcome_persisted": True,
        },
        "input_output_sha256": {
            UNSEAL_NAME: marker_sha, ANALYSIS_NAME: analysis_sha,
            TABLE_NAME: table_sha, BOOTSTRAP_NAME: bootstrap_sha, SUMMARY_NAME: summary_sha,
        },
        "status": "VERIFIED_READY_FOR_PLANNING_AUDIT",
    }
    verifier_sha = _write_new_json(GATE_D_ROOT / VERIFIER_NAME, verifier)
    terminal = {
        "schema_version": "loopscope.phase5.gate-d-terminal-receipt.v1",
        "gate": GATE, "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID, "created_at_utc": utc_now(),
        "git": git, "unseal_count": 1, "analysis_count": 1,
        "verifier_result": "PASS",
        "terminal_label": TERMINAL_LABEL,
        "window_decision": "ABSTAIN_NO_POINT_ELIGIBLE",
        "selected_window": None, "eligible_count": 0, "selection_frequency": None,
        "output_sha256_excluding_self": {
            UNSEAL_NAME: marker_sha, ANALYSIS_NAME: analysis_sha,
            TABLE_NAME: table_sha, BOOTSTRAP_NAME: bootstrap_sha,
            VERIFIER_NAME: verifier_sha, SUMMARY_NAME: summary_sha,
        },
        "safety": {
            "gate_c_rerun": False, "model_loaded": False, "gpu_used": False,
            "slurm_used": False, "selector_or_panel_changed": False,
            "second_unseal_or_analysis": False, "prior_artifact_modified": False,
        },
        "status": "GATE_D_READY_FOR_PLANNING_AUDIT",
        "requested_decision": "PASS",
    }
    terminal_sha = _write_new_json(GATE_D_ROOT / TERMINAL_NAME, terminal)
    return {
        "terminal_label": TERMINAL_LABEL,
        "window_decision": "ABSTAIN_NO_POINT_ELIGIBLE",
        "unseal_count": 1, "analysis_count": 1, "verifier_result": "PASS",
        "high3_minus_low3": analysis["groups"]["high3_minus_low3"],
        "blind_six_selector_score_vs_gain": analysis["blind_six_selector_score_vs_gain"],
        "artifact_sha256": {**terminal["output_sha256_excluding_self"], TERMINAL_NAME: terminal_sha},
    }


__all__ = [
    "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED", "GATE_D_ROOT", "Phase5GateDError",
    "build_statistical_analysis", "exact_mcnemar_p", "exact_spearman",
    "git_provenance", "linear_percentile", "run_one_shot", "spearman_rho",
    "subject_stratified_joint_bootstrap", "validate_inputs",
]
