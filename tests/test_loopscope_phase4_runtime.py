from __future__ import annotations

import unittest

from tflt.loopscope.phase4_runtime import (
    CATEGORIES,
    Phase4RuntimeError,
    deterministic_option_permutation,
    permute_safe_target,
    project_target_row,
    proportional_category_quotas,
    render_exact_prefix,
    template_safe_targets,
    tokenization_closure,
    validation_demos_by_category,
)


class _Renderer:
    @staticmethod
    def fewshot_to_text(doc):
        return "D:%s:%s\n\n" % (doc["question"], doc["cot_content"])

    @staticmethod
    def doc_to_text(doc):
        return "Question:\n%s\nOptions:\n%s\nAnswer: Let's think step by step." % (
            doc["question"],
            "|".join(doc["options"]),
        )


def _safe_target():
    return {
        "question_id": 7,
        "category": "biology",
        "src": "fixture",
        "question": "safe question",
        "ordered_options": ["option-%d" % index for index in range(10)],
    }


class Phase4RuntimeTests(unittest.TestCase):
    def test_projector_requires_prior_allowlist_projection(self) -> None:
        projected = {
            "question_id": 7,
            "category": "biology",
            "src": "fixture",
            "question": "safe question",
            "options": ["option-%d" % index for index in range(10)],
        }
        self.assertEqual(project_target_row(projected)["ordered_options"], projected["options"])
        raw = dict(projected, answer="A", cot_content="sealed")
        with self.assertRaisesRegex(Phase4RuntimeError, "allowlist"):
            project_target_row(raw)

    def test_renderer_uses_exact_five_demo_order_and_gold_free_target(self) -> None:
        validation = []
        for category in CATEGORIES:
            for index in range(5):
                validation.append(
                    {
                        "category": category,
                        "question": "%s-%d" % (category, index),
                        "options": ["x"] * 10,
                        "cot_content": "demo-%d" % index,
                    }
                )
        demos = validation_demos_by_category(validation)
        prefix = render_exact_prefix(_safe_target(), demos["biology"], _Renderer)
        self.assertEqual(prefix.count("D:biology-"), 5)
        self.assertTrue(prefix.endswith("Answer: Let's think step by step."))
        self.assertNotIn("answer", _safe_target())

    def test_quota_and_identity_only_permutation_are_deterministic(self) -> None:
        counts = {category: 100 for category in CATEGORIES}
        quotas = proportional_category_quotas(counts, 512)
        self.assertEqual(sum(quotas.values()), 512)
        first = deterministic_option_permutation("identity")
        self.assertEqual(first, deterministic_option_permutation("identity"))
        self.assertNotEqual(first, list(range(10)))
        permuted = permute_safe_target(_safe_target(), first)
        self.assertCountEqual(permuted["ordered_options"], _safe_target()["ordered_options"])
        with self.assertRaisesRegex(Phase4RuntimeError, "exceeds"):
            proportional_category_quotas(counts, 2000)

    def test_template_controls_are_exact_14_and_safe(self) -> None:
        templates = template_safe_targets()
        self.assertEqual([row["category"] for row in templates], list(CATEGORIES))
        self.assertEqual(len(templates), 14)
        self.assertTrue(all(len(row["ordered_options"]) == 10 for row in templates))

    def test_tokenization_closure_keeps_last_token_hash(self) -> None:
        metadata = {
            "rendered_prefix_sha256": "a" * 64,
            "rendered_token_ids_sha256": "b" * 64,
            "attention_mask_sha256": "c" * 64,
            "last_effective_prefix_token_index": 9,
            "last_effective_prefix_token_sha256": "d" * 64,
            "sequence_length": 10,
        }
        closure = tokenization_closure(metadata)
        self.assertEqual(closure["last_effective_prefix_token_sha256"], "d" * 64)
        self.assertNotIn("sequence_length", closure)
        broken = dict(metadata)
        broken.pop("last_effective_prefix_token_sha256")
        with self.assertRaisesRegex(Phase4RuntimeError, "lacks"):
            tokenization_closure(broken)


if __name__ == "__main__":
    unittest.main()
