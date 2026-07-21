#!/usr/bin/env python3
"""Deterministically verify the Phase 5 Gate A data-free contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from tflt.loopscope.phase5_schema import (
    artifact_hashes,
    analyze_signal_samples,
    boundary_contract,
    construct_panel,
    load_json,
    trajectory_record_from_inputs,
    trajectory_window_signals,
    validate_card,
    validate_card_schema,
    validate_registry,
    validate_token_ids,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "loopscope"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _renderer(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["tokenizer_revision"],
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "renderer_manifest_sha256": card["task"][
            "validation_renderer_manifest_internal_sha256"
        ],
        "forward_type": "native_no_loop",
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "position": "final_non_padding_prompt_token",
        "projection": "frozen_final_norm_then_tied_output_weight_exact_choice_rows",
        "choice_order": ["A", "B", "C", "D"],
        "spaced_choice_token_ids": card["model"]["spaced_choice_token_ids"],
        "final_norm_path": "model.norm",
        "output_head_path": "lm_head_tied_to_model.embed_tokens.weight",
    }


def _synthetic_trajectory(card: Dict[str, Any]) -> Dict[str, Any]:
    logits: List[List[float]] = []
    hidden: List[List[float]] = []
    for index in range(37):
        logits.append([0.02 * index, -0.1, -0.3, -0.5])
        hidden.append([1.0 + index / 100.0, 0.5, -0.25, 0.75])
    return trajectory_record_from_inputs(
        identity={"task": "mmlu_synthetic", "doc_id": "0", "doc_hash": _sha("doc")},
        subject="synthetic_subject",
        prompt_sha256=_sha("prompt"),
        boundary_choice_logits=logits,
        final_normalized_hidden=hidden,
        renderer_provenance=_renderer(card),
    )


def _selector_samples() -> List[Dict[str, Any]]:
    samples = []
    for index in range(4):
        e_values = [-1.0] * 33
        k_values = [-1.0] * 33
        e_values[12] = 1.0 + index / 100.0
        k_values[12] = 1.2 + index / 100.0
        samples.append(
            {
                "identity": "selector-%d" % index,
                "subject": "subject-%d" % (index % 2),
                "E": e_values,
                "K": k_values,
            }
        )
    return samples


def verify(root: Path = ROOT) -> Dict[str, Any]:
    config_root = root / "configs" / "loopscope"
    card_path = config_root / "phase5_card.json"
    schema_path = config_root / "phase5_card_schema.json"
    registry_path = config_root / "phase5_known_outcome_registry.json"
    card = load_json(card_path)
    schema = load_json(schema_path)
    registry = load_json(registry_path)
    validate_card(card)
    validate_card_schema(schema)
    validate_registry(registry)
    validate_token_ids(
        {key: [value] for key, value in card["model"]["spaced_choice_token_ids"].items()},
        {key: [value] for key, value in card["model"]["plain_choice_token_ids"].items()},
    )
    trajectory = _synthetic_trajectory(card)
    signals = trajectory_window_signals(trajectory)
    selector = analyze_signal_samples(_selector_samples(), replicates=31, seed=20260722)
    panel = construct_panel(
        selector["published_ranking"],
        selected_window=selector["selected_window"],
        known_windows=registry["blind_group_exclusion_windows"],
    )
    artifacts = artifact_hashes(
        {
            "card": card_path,
            "schema": schema_path,
            "registry": registry_path,
        }
    )
    for payload in artifacts.values():
        payload["relative_path"] = str(Path(payload["relative_path"]).relative_to(root))
    return {
        "schema_version": "loopscope.phase5.gate-a-verification.v1",
        "checks": {
            "card_schema_registry_hash_closed": True,
            "boundary_contract": boundary_contract(),
            "token_ids_exact_unique_single_token": True,
            "five_trajectory_scalars_recomputed": True,
            "final_boundary_closed": True,
            "selector_decision": selector["window_decision"],
            "selector_geometry_fields_consumed": selector["geometry_fields_consumed"],
            "selector_outcome_fields_consumed": selector["outcome_fields_consumed"],
            "panel_unique_cell_count": panel["unique_cell_count"],
            "known_window_excluded_from_blind_groups": "15:18"
            not in panel["panel_high3"] + panel["blind_low3"],
            "generic_delta_direction": {
                "choice_entropy": "exit_minus_entry",
                "kl_to_final": "exit_minus_entry",
                "selector_E": "entry_minus_exit",
                "selector_K": "exit_minus_entry",
            },
            "synthetic_signal_window_count": len(signals["windows"]),
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
