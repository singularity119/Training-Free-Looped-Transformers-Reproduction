import unittest

from tflt.loopscope.phase6_anchor import (
    ANSWER_REGEX_PATTERN,
    ANSWER_SPAN_EXTRACTOR_SHA256,
    DEFAULT_TEMPLATE_YAML_SHA256,
    GENERATED_ID_TEXT_ALIGNER_SHA256,
    LM_EVAL_UTILS_SHA256,
    LM_EVAL_VERSION,
    MMLU_PRO_YAML_SHA256,
    AnswerSpanError,
    AnswerSpan,
    AnchorEligibilityDecision,
    GeneratedTextAlignmentError,
    NoPrecedingGeneratedTokenError,
    ReplayPrefixError,
    answer_span_extractor,
    classify_anchor_eligibility,
    generated_id_text_aligner,
    replay_prefix_ids,
)


class FakeTokenizer:
    def __init__(self, pieces, special_ids=()):
        self.pieces = dict(pieces)
        self.all_special_ids = list(special_ids)

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del clean_up_tokenization_spaces
        return "".join(
            ""
            if skip_special_tokens and token_id in self.all_special_ids
            else self.pieces[token_id]
            for token_id in token_ids
        )


class PrefixRewritingTokenizer(FakeTokenizer):
    """Characterize context-sensitive prefix decoding without changing full decode."""

    def __init__(self, pieces, rewrites, special_ids=()):
        super().__init__(pieces, special_ids=special_ids)
        self.rewrites = {tuple(key): value for key, value in rewrites.items()}

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        key = tuple(token_ids)
        if key in self.rewrites:
            return self.rewrites[key]
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


class Phase6AnswerSpanTests(unittest.TestCase):
    def test_identity_is_bound_to_exact_source_bundle(self):
        self.assertEqual(LM_EVAL_VERSION, "0.4.11")
        self.assertEqual(
            LM_EVAL_UTILS_SHA256,
            "74ab409c4e4c96e4351fbe6122d519f64dd3e11381a676957631e858923cc9fc",
        )
        self.assertEqual(
            MMLU_PRO_YAML_SHA256,
            "0271e3fdbbb0e8df5b6909786600c1292da29842086b521a1181954941156f94",
        )
        self.assertEqual(
            DEFAULT_TEMPLATE_YAML_SHA256,
            "356e937a958288fafd8f07b03da7fd4825977e9bb3d62136931c9f649b600647",
        )
        self.assertEqual(
            ANSWER_REGEX_PATTERN, r"answer is \(?([ABCDEFGHIJ])\)?"
        )
        self.assertEqual(len(ANSWER_SPAN_EXTRACTOR_SHA256), 64)
        self.assertEqual(len(GENERATED_ID_TEXT_ALIGNER_SHA256), 64)

    def test_unique_parenthesized_answer_preserves_character_and_byte_span(self):
        text = "Reasoning. answer is (C)."
        span = answer_span_extractor(text)
        self.assertEqual(text[slice(*span.char_span)], "C")
        self.assertEqual(text.encode("utf-8")[slice(*span.byte_span)], b"C")
        self.assertEqual(span.answer_match_count, 1)
        self.assertEqual(span.selected_match_ordinal, 0)

    def test_missing_answer_fails_closed(self):
        with self.assertRaises(AnswerSpanError):
            answer_span_extractor("The result might be C.")

    def test_corrected_multiple_matches_selects_first(self):
        text = "answer is A; correction: answer is B"
        span = answer_span_extractor(text)
        self.assertEqual(span.char_span, (10, 11))
        self.assertEqual(span.answer_match_count, 2)
        self.assertEqual(span.selected_match_ordinal, 0)

    def test_repeated_identical_matches_selects_first(self):
        text = "answer is D; therefore answer is D"
        span = answer_span_extractor(text)
        self.assertEqual(span.char_span, (10, 11))
        self.assertEqual(span.answer_match_count, 2)
        self.assertEqual(span.selected_match_ordinal, 0)

    def test_regex_is_case_sensitive(self):
        with self.assertRaises(AnswerSpanError):
            answer_span_extractor("Answer is A")
        span = answer_span_extractor("Answer is A; final answer is B")
        self.assertEqual(span.char_span, (29, 30))


class Phase6GeneratedAlignmentTests(unittest.TestCase):
    def test_seven_preserved_gate_d_alignment_normal_paths_resolve(self):
        base_pieces = {
            1: "reason",
            2: "ing ",
            3: "answer is ",
            4: "A",
            5: ".",
        }
        cases = [
            PrefixRewritingTokenizer(base_pieces, {(1,): "\ufffd"}),
            PrefixRewritingTokenizer(base_pieces, {(1, 2): "reasoning\ufffd"}),
            PrefixRewritingTokenizer(base_pieces, {(1,): ""}),
            PrefixRewritingTokenizer(base_pieces, {(1, 2): "reason"}),
            PrefixRewritingTokenizer(base_pieces, {(1,): "reason\ufffd"}),
            FakeTokenizer({**base_pieces, 90: "<|im_end|>"}, special_ids={90}),
            FakeTokenizer(
                {**base_pieces, 90: "<|im_end|>", 91: "<|endoftext|>"},
                special_ids={90, 91},
            ),
        ]
        generated_id_cases = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 90],
            [1, 2, 3, 4, 5, 90, 91],
        ]
        for case_index, (tokenizer, generated_ids) in enumerate(
            zip(cases, generated_id_cases)
        ):
            with self.subTest(case_index=case_index):
                text = tokenizer.decode(generated_ids)
                alignment = generated_id_text_aligner(
                    generated_ids, tokenizer, text
                )
                self.assertEqual(alignment.answer_first_token_index, 3)
                self.assertEqual(alignment.probe_token_index, 2)

    def test_answer_span_inside_token(self):
        tokenizer = FakeTokenizer({1: "thinking ", 2: "answer is (C)."})
        text = tokenizer.decode([1, 2])
        alignment = generated_id_text_aligner([1, 2], tokenizer, text)
        self.assertEqual(alignment.answer_first_token_index, 1)
        self.assertEqual(alignment.probe_token_index, 0)

    def test_half_open_boundary_belongs_to_token_on_right(self):
        tokenizer = FakeTokenizer({1: "answer is ", 2: "D", 3: "."})
        text = tokenizer.decode([1, 2, 3])
        alignment = generated_id_text_aligner([1, 2, 3], tokenizer, text)
        self.assertEqual(alignment.answer_first_token_index, 1)
        self.assertEqual(
            alignment.answer_span.byte_start,
            alignment.token_byte_spans[1].byte_start,
        )

    def test_unicode_uses_utf8_byte_offsets(self):
        tokenizer = FakeTokenizer({1: "理由é: ", 2: "answer is ", 3: "E"})
        text = tokenizer.decode([1, 2, 3])
        alignment = generated_id_text_aligner([1, 2, 3], tokenizer, text)
        self.assertEqual(alignment.answer_first_token_index, 2)
        self.assertGreater(
            alignment.answer_span.byte_start, alignment.answer_span.char_start
        )
        self.assertEqual(
            text.encode("utf-8")[
                alignment.answer_span.byte_start : alignment.answer_span.byte_end
            ],
            b"E",
        )

    def test_visible_special_token_content_is_forbidden(self):
        tokenizer = FakeTokenizer(
            {1: "reason ", 99: "answer is A"}, special_ids={99}
        )
        text = tokenizer.decode([1, 99])
        with self.assertRaises(GeneratedTextAlignmentError):
            generated_id_text_aligner([1, 99], tokenizer, text)

    def test_full_text_round_trip_is_exact(self):
        tokenizer = FakeTokenizer({1: "reason ", 2: "answer is B"})
        with self.assertRaises(GeneratedTextAlignmentError):
            generated_id_text_aligner(
                [1, 2], tokenizer, "reason  answer is B"
            )

    def test_caller_cannot_bypass_frozen_regex_with_forged_span(self):
        tokenizer = FakeTokenizer({1: "reason ", 2: "answer is B"})
        text = tokenizer.decode([1, 2])
        real_span = answer_span_extractor(text)
        forged_span = AnswerSpan(
            char_start=real_span.char_start - 1,
            char_end=real_span.char_end,
            byte_start=real_span.byte_start - 1,
            byte_end=real_span.byte_end,
            answer_match_count=real_span.answer_match_count,
            selected_match_ordinal=real_span.selected_match_ordinal,
        )
        with self.assertRaisesRegex(
            GeneratedTextAlignmentError, "frozen regex capture"
        ):
            generated_id_text_aligner(
                [1, 2], tokenizer, text, answer_span=forged_span
            )

    def test_first_generated_token_has_no_preceding_probe(self):
        tokenizer = FakeTokenizer({1: "answer is A"})
        with self.assertRaisesRegex(
            NoPrecedingGeneratedTokenError, "no preceding generated probe"
        ):
            generated_id_text_aligner(
                [1], tokenizer, tokenizer.decode([1])
            )

    def test_visible_special_before_selected_answer_obeys_full_decode_path(self):
        tokenizer = FakeTokenizer(
            {90: "<|im_end|>", 1: "reason ", 2: "answer is A"},
            special_ids={90},
        )
        text = tokenizer.decode([90, 1, 2])
        alignment = generated_id_text_aligner([90, 1, 2], tokenizer, text)
        self.assertEqual(alignment.answer_first_token_index, 2)
        self.assertEqual(alignment.probe_token_index, 1)

    def test_probe_is_immediately_before_answer_token(self):
        tokenizer = FakeTokenizer(
            {1: "reason", 2: "ing. ", 3: "answer is ", 4: "F"}
        )
        text = tokenizer.decode([1, 2, 3, 4])
        alignment = generated_id_text_aligner([1, 2, 3, 4], tokenizer, text)
        self.assertEqual(alignment.answer_first_token_index, 3)
        self.assertEqual(alignment.probe_token_index, 2)

    def test_multiple_match_alignment_and_replay_use_ordinal_zero(self):
        tokenizer = FakeTokenizer(
            {
                1: "reason ",
                2: "answer is ",
                3: "A",
                4: "; correction: answer is ",
                5: "B",
            }
        )
        generated_ids = [1, 2, 3, 4, 5]
        text = tokenizer.decode(generated_ids)
        alignment = generated_id_text_aligner(generated_ids, tokenizer, text)
        self.assertEqual(alignment.answer_span.answer_match_count, 2)
        self.assertEqual(alignment.answer_span.selected_match_ordinal, 0)
        self.assertEqual(alignment.answer_first_token_index, 2)
        self.assertEqual(alignment.probe_token_index, 1)
        replay = replay_prefix_ids([10, 11], generated_ids, 2)
        self.assertEqual(replay, (10, 11, 1, 2))


class Phase6ReplayPrefixTests(unittest.TestCase):
    def test_replay_prefix_is_prompt_plus_ids_strictly_before_answer(self):
        expected = replay_prefix_ids(
            prompt_ids=[10, 11],
            generated_ids=[20, 21, 22],
            answer_first_token_index=2,
        )
        self.assertEqual(expected, (10, 11, 20, 21))
        self.assertEqual(
            replay_prefix_ids([10, 11], [20, 21, 22], 2, expected),
            expected,
        )

    def test_replay_id_mismatch_fails_exactly(self):
        with self.assertRaisesRegex(ReplayPrefixError, "exactly match"):
            replay_prefix_ids(
                [10, 11], [20, 21, 22], 2, [10, 11, 20, 99]
            )

    def test_replay_requires_preceding_generated_token(self):
        with self.assertRaisesRegex(ReplayPrefixError, "preceding generated"):
            replay_prefix_ids([10], [20, 21], 0)

    def test_answer_index_must_reference_an_existing_generated_id(self):
        with self.assertRaisesRegex(ReplayPrefixError, "out of range"):
            replay_prefix_ids([10], [20, 21], 2)


class Phase6AnchorEligibilityTests(unittest.TestCase):
    def test_zero_match_is_not_expressed_without_fallback(self):
        tokenizer = FakeTokenizer({1: "The result might be C."})
        decision = classify_anchor_eligibility(
            [1], tokenizer, tokenizer.decode([1])
        )
        self.assertIsInstance(decision, AnchorEligibilityDecision)
        self.assertEqual(decision.state, "ANCHOR_NOT_EXPRESSED")
        self.assertIsNone(decision.answer_span)
        self.assertIsNone(decision.alignment)

    def test_first_token_match_is_not_expressed(self):
        tokenizer = FakeTokenizer({1: "answer is A"})
        decision = classify_anchor_eligibility(
            [1], tokenizer, tokenizer.decode([1])
        )
        self.assertEqual(decision.state, "ANCHOR_NOT_EXPRESSED")
        self.assertIsNotNone(decision.answer_span)
        self.assertIsNone(decision.alignment)

    def test_match_present_alignment_error_is_not_masked(self):
        tokenizer = FakeTokenizer({1: "reason ", 2: "answer is B"})
        with self.assertRaisesRegex(
            GeneratedTextAlignmentError, "round-trip"
        ):
            classify_anchor_eligibility(
                [1, 2], tokenizer, "reason  answer is B"
            )


if __name__ == "__main__":
    unittest.main()
