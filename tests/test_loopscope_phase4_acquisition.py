from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.phase4_acquisition import (
    P4BAcquisitionError,
    SHARD_RECEIPT_SCHEMA,
    build_shard_manifest,
    extract_known_outcome_windows,
    merge_completed_shards,
    require_non_degenerate_width4_scores,
    validate_shard_manifest,
    wrap_outcome_panel,
    write_new_json,
    write_new_jsonl,
)
from tflt.loopscope.phase4_schema import file_sha256, load_json_object, semantic_sha256
from tflt.loopscope.phase4_selector import analyze_selector, build_outcome_panel
from tflt.loopscope.phase4_verifier import _synthetic_scalar_record, _synthetic_source_record


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/phase4_pv_ek_trs_card.json"


def _membership(count: int):
    return [
        {"canonical_identity": "identity-%03d" % index, "category": "category"}
        for index in range(count)
    ]


def _trajectory(identity: str, index: int):
    record = copy.deepcopy(_synthetic_scalar_record(index))
    record["canonical_identity"] = identity
    return record


def _write_completed_shards(root: Path, manifest, *, order=None):
    order = list(range(manifest["shard_count"])) if order is None else list(order)
    for shard_index in order:
        shard = manifest["shards"][shard_index]
        output = root / shard["output_relative"]
        records = [
            _trajectory(member["canonical_identity"], member["ordinal"])
            for member in shard["records"]
        ]
        record_path = output / "records.jsonl"
        write_new_jsonl(record_path, records)
        receipt = {
            "schema_version": SHARD_RECEIPT_SCHEMA,
            "status": "COMPLETED",
            "shard_id": shard["shard_id"],
            "membership_sha256": shard["membership_sha256"],
            "record_count": len(records),
            "record_file_sha256": file_sha256(record_path),
            "ordered_record_sha256": semantic_sha256(records),
        }
        receipt["manifest_sha256"] = semantic_sha256(receipt)
        write_new_json(output / "receipt.json", receipt)


class Phase4AcquisitionTests(unittest.TestCase):
    def test_round_robin_manifest_is_deterministic_exact_union_for_all_kinds(self) -> None:
        records = _membership(11)
        for kind in ("original", "option", "template"):
            with self.subTest(kind=kind):
                manifest = build_shard_manifest(
                    records,
                    kind=kind,
                    shard_count=3,
                    run_root=Path("/tmp/p4b-unit"),
                    bindings={"card_sha256": "a" * 64},
                )
                validate_shard_manifest(manifest, records)
                ordinals = [
                    member["ordinal"]
                    for shard in manifest["shards"]
                    for member in shard["records"]
                ]
                self.assertEqual(sorted(ordinals), list(range(11)))
                self.assertEqual(len(ordinals), len(set(ordinals)))
                self.assertEqual(
                    manifest,
                    build_shard_manifest(
                        records,
                        kind=kind,
                        shard_count=3,
                        run_root=Path("/tmp/p4b-unit"),
                        bindings={"card_sha256": "a" * 64},
                    ),
                )

    def test_merge_restores_expected_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = _membership(8)
            manifest = build_shard_manifest(
                records, kind="original", shard_count=3, run_root=root, bindings={}
            )
            _write_completed_shards(root, manifest, order=(2, 0, 1))
            merged = merge_completed_shards(
                run_root=root,
                shard_manifest=manifest,
                expected_records=records,
                merged_jsonl_relative="original/merged.jsonl",
                merged_manifest_relative="original/merged_manifest.json",
            )
            identities = [
                item["canonical_identity"]
                for item in __import__(
                    "tflt.loopscope.phase4_acquisition", fromlist=["load_strict_jsonl"]
                ).load_strict_jsonl(root / "original/merged.jsonl")
            ]
            self.assertEqual(identities, [item["canonical_identity"] for item in records])
            self.assertEqual(merged["record_count"], 8)

    def test_merge_rejects_partial_duplicate_and_hash_drift(self) -> None:
        cases = ("partial", "duplicate", "hash")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records = _membership(6)
                manifest = build_shard_manifest(
                    records, kind="original", shard_count=2, run_root=root, bindings={}
                )
                _write_completed_shards(root, manifest)
                if case == "partial":
                    (root / manifest["shards"][1]["output_relative"] / "receipt.json").unlink()
                elif case == "duplicate":
                    record_path = root / manifest["shards"][1]["output_relative"] / "records.jsonl"
                    shard_records = __import__(
                        "tflt.loopscope.phase4_acquisition", fromlist=["load_strict_jsonl"]
                    ).load_strict_jsonl(record_path)
                    shard_records[0]["canonical_identity"] = manifest["shards"][0]["records"][0][
                        "canonical_identity"
                    ]
                    record_path.unlink()
                    write_new_jsonl(record_path, shard_records)
                    receipt_path = record_path.with_name("receipt.json")
                    receipt = load_json_object(receipt_path)
                    receipt_path.unlink()
                    receipt["record_file_sha256"] = file_sha256(record_path)
                    receipt["ordered_record_sha256"] = semantic_sha256(shard_records)
                    receipt["manifest_sha256"] = semantic_sha256(
                        {key: value for key, value in receipt.items() if key != "manifest_sha256"}
                    )
                    write_new_json(receipt_path, receipt)
                else:
                    receipt_path = root / manifest["shards"][0]["output_relative"] / "receipt.json"
                    receipt = load_json_object(receipt_path)
                    receipt_path.unlink()
                    receipt["record_file_sha256"] = "f" * 64
                    receipt["manifest_sha256"] = semantic_sha256(
                        {key: value for key, value in receipt.items() if key != "manifest_sha256"}
                    )
                    write_new_json(receipt_path, receipt)
                with self.assertRaises(P4BAcquisitionError):
                    merge_completed_shards(
                        run_root=root,
                        shard_manifest=manifest,
                        expected_records=records,
                        merged_jsonl_relative="original/merged.jsonl",
                        merged_manifest_relative="original/merged_manifest.json",
                    )

    def test_formal_original_source_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = [_synthetic_source_record(index) for index in range(4)]
            records = [
                {"canonical_identity": source["canonical_identity"], "category": source["category"]}
                for source in sources
            ]
            manifest = build_shard_manifest(
                records, kind="original", shard_count=2, run_root=root, bindings={}
            )
            _write_completed_shards(root, manifest)
            for shard in manifest["shards"]:
                record_path = root / shard["output_relative"] / "records.jsonl"
                shard_records = __import__(
                    "tflt.loopscope.phase4_acquisition", fromlist=["load_strict_jsonl"]
                ).load_strict_jsonl(record_path)
                for trajectory, member in zip(shard_records, shard["records"]):
                    source = sources[member["ordinal"]]
                    trajectory["category"] = source["category"]
                    trajectory["producer_provenance"]["source_closure"] = {
                        key: source[key]
                        for key in (
                            "canonical_identity",
                            "sanitized_content_sha256",
                            "rendered_prefix_sha256",
                            "rendered_token_ids_sha256",
                            "attention_mask_sha256",
                        )
                    }
                record_path.unlink()
                write_new_jsonl(record_path, shard_records)
                receipt_path = record_path.with_name("receipt.json")
                receipt = load_json_object(receipt_path)
                receipt_path.unlink()
                receipt["record_file_sha256"] = file_sha256(record_path)
                receipt["ordered_record_sha256"] = semantic_sha256(shard_records)
                receipt["manifest_sha256"] = semantic_sha256(
                    {key: value for key, value in receipt.items() if key != "manifest_sha256"}
                )
                write_new_json(receipt_path, receipt)
            merged = merge_completed_shards(
                run_root=root,
                shard_manifest=manifest,
                expected_records=records,
                source_records=sources,
                require_full_population=False,
                merged_jsonl_relative="original/merged.jsonl",
                merged_manifest_relative="original/merged_manifest.json",
            )
            self.assertTrue(merged["source_identity_closure_verified"])

    def test_all_tie_scores_are_rejected(self) -> None:
        flat = []
        for index in range(12):
            record = copy.deepcopy(_synthetic_scalar_record(index))
            for boundary in record["boundaries"]:
                boundary["full_vocabulary_entropy"] = 5.0
                boundary["normalized_entropy"] = 0.5
                boundary["kl_to_final"] = 0.0
            flat.append(record)
        report = analyze_selector(flat, replicates=16, seed=20260717)
        with self.assertRaisesRegex(P4BAcquisitionError, "tied/degenerate"):
            require_non_degenerate_width4_scores(report)

    def test_card_registry_and_fresh_panel_flags(self) -> None:
        card = load_json_object(CARD)
        records = [_synthetic_scalar_record(index) for index in range(24)]
        report = analyze_selector(records, replicates=16, seed=20260717)
        known = extract_known_outcome_windows(card)
        self.assertEqual(known, ["15:18"])
        panel = build_outcome_panel(report, known)
        wrapped = wrap_outcome_panel(
            panel,
            card=card,
            selector_freeze_sha256="a" * 64,
            bindings={"card_sha256": "b" * 64},
        )
        self.assertIs(wrapped["fresh_baseline"], True)
        self.assertIs(wrapped["fresh_15_18"], True)
        self.assertFalse(wrapped["panel"]["variable_width_outcomes_authorized"])


if __name__ == "__main__":
    unittest.main()
