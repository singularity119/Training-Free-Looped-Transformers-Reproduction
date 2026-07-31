"""Pure-Python Phase 6 final-answer anchor resolution.

The extractor mirrors lm-eval's ``custom-extract -> take_first`` path: it
enumerates every capture-group span in generated-text order and selects ordinal
zero.  Gate D-2 classifies zero matches and a selected answer carried by the
first generated token as ``ANCHOR_NOT_EXPRESSED``.  All other alignment
failures remain fail-closed engineering errors.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple


LM_EVAL_VERSION = "0.4.11"
LM_EVAL_UTILS_SHA256 = (
    "74ab409c4e4c96e4351fbe6122d519f64dd3e11381a676957631e858923cc9fc"
)
MMLU_PRO_YAML_SHA256 = (
    "0271e3fdbbb0e8df5b6909786600c1292da29842086b521a1181954941156f94"
)
DEFAULT_TEMPLATE_YAML_SHA256 = (
    "356e937a958288fafd8f07b03da7fd4825977e9bb3d62136931c9f649b600647"
)

ANSWER_REGEX_PATTERN = r"answer is \(?([ABCDEFGHIJ])\)?"
ANSWER_REGEX = re.compile(ANSWER_REGEX_PATTERN)

ANSWER_SPAN_EXTRACTOR_VERSION = "loopscope-phase6-answer-span-v2"
GENERATED_ID_TEXT_ALIGNER_VERSION = "loopscope-phase6-generated-id-aligner-v2"
REPLAY_PREFIX_VERSION = "loopscope-phase6-replay-prefix-v1"

ANCHOR_ELIGIBLE = "ANCHOR_ELIGIBLE"
ANCHOR_NOT_EXPRESSED = "ANCHOR_NOT_EXPRESSED"

# These kwargs are part of the aligner's identity.  Callers may pass a
# different frozen mapping explicitly, but must then retain that mapping with
# the resulting artifact.
DEFAULT_DECODE_KWARGS: Mapping[str, Any] = MappingProxyType(
    {
        "skip_special_tokens": False,
        "clean_up_tokenization_spaces": False,
    }
)


def _identity_hash(component: str, version: str, **extra: Any) -> str:
    payload = {
        "component": component,
        "version": version,
        "lm_eval_version": LM_EVAL_VERSION,
        "lm_eval_utils_sha256": LM_EVAL_UTILS_SHA256,
        "mmlu_pro_yaml_sha256": MMLU_PRO_YAML_SHA256,
        "default_template_yaml_sha256": DEFAULT_TEMPLATE_YAML_SHA256,
        "answer_regex_pattern": ANSWER_REGEX_PATTERN,
        **extra,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


ANSWER_SPAN_EXTRACTOR_SHA256 = _identity_hash(
    "answer_span_extractor",
    ANSWER_SPAN_EXTRACTOR_VERSION,
    match_policy="all_capture_group_1_spans_select_ordinal_0",
)
GENERATED_ID_TEXT_ALIGNER_SHA256 = _identity_hash(
    "generated_id_text_aligner",
    GENERATED_ID_TEXT_ALIGNER_VERSION,
    span_unit="utf8_bytes",
    boundary_policy="half_open_boundary_belongs_right",
    decode_kwargs=dict(DEFAULT_DECODE_KWARGS),
    full_text_round_trip="exact",
    token_mapping="unique_stable_adjacent_prefix_pair",
    unstable_intermediate_prefixes="allowed_outside_selected_token_boundary",
    special_token_policy="carrier_forbidden_others_allowed_if_full_closure",
)
REPLAY_PREFIX_SHA256 = _identity_hash(
    "replay_prefix",
    REPLAY_PREFIX_VERSION,
    prefix_policy="prompt_plus_generated_strictly_before_answer_token",
    replay_match="exact_ids",
)

SOURCE_BUNDLE_IDENTITY: Mapping[str, str] = MappingProxyType(
    {
        "lm_eval_version": LM_EVAL_VERSION,
        "lm_eval_utils_sha256": LM_EVAL_UTILS_SHA256,
        "mmlu_pro_yaml_sha256": MMLU_PRO_YAML_SHA256,
        "default_template_yaml_sha256": DEFAULT_TEMPLATE_YAML_SHA256,
        "answer_regex_pattern": ANSWER_REGEX_PATTERN,
        "answer_span_extractor_version": ANSWER_SPAN_EXTRACTOR_VERSION,
        "answer_span_extractor_sha256": ANSWER_SPAN_EXTRACTOR_SHA256,
        "generated_id_text_aligner_version": GENERATED_ID_TEXT_ALIGNER_VERSION,
        "generated_id_text_aligner_sha256": GENERATED_ID_TEXT_ALIGNER_SHA256,
        "replay_prefix_version": REPLAY_PREFIX_VERSION,
        "replay_prefix_sha256": REPLAY_PREFIX_SHA256,
    }
)


class AnchorResolutionError(ValueError):
    """Base class for fail-closed Phase 6 anchor failures."""


class AnswerSpanError(AnchorResolutionError):
    """The frozen answer regex did not produce any capture span."""


class GeneratedTextAlignmentError(AnchorResolutionError):
    """Generated IDs and text could not be aligned exactly."""


class NoPrecedingGeneratedTokenError(AnchorResolutionError):
    """The selected answer is carried by the first generated token."""


class ReplayPrefixError(AnchorResolutionError):
    """The replay prefix was invalid or did not match the frozen IDs."""


@dataclass(frozen=True)
class AnswerSpan:
    """The selected capture-group-1 span and sanitized match metadata."""

    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    answer_match_count: int
    selected_match_ordinal: int

    @property
    def char_span(self) -> Tuple[int, int]:
        return (self.char_start, self.char_end)

    @property
    def byte_span(self) -> Tuple[int, int]:
        return (self.byte_start, self.byte_end)


@dataclass(frozen=True)
class TokenByteSpan:
    """Half-open UTF-8 byte interval contributed by one generated ID."""

    generated_index: int
    byte_start: int
    byte_end: int
    is_special: bool

    @property
    def byte_span(self) -> Tuple[int, int]:
        return (self.byte_start, self.byte_end)


@dataclass(frozen=True)
class GeneratedAnswerAlignment:
    """Resolved generated-token location for the selected answer span."""

    answer_span: AnswerSpan
    answer_first_token_index: int
    probe_token_index: int
    answer_token_byte_span: Tuple[int, int]
    token_byte_spans: Tuple[TokenByteSpan, ...]
    extractor_sha256: str = ANSWER_SPAN_EXTRACTOR_SHA256
    aligner_sha256: str = GENERATED_ID_TEXT_ALIGNER_SHA256


@dataclass(frozen=True)
class AnchorEligibilityDecision:
    """Outcome-blind Gate D-2 anchor eligibility classification."""

    state: str
    answer_span: Optional[AnswerSpan]
    alignment: Optional[GeneratedAnswerAlignment]


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError("%s must be str" % name)
    return value


def _token_id_tuple(values: Sequence[int], name: str) -> Tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError("%s must be a sequence of integer token IDs" % name) from exc

    result = []
    for value in raw_values:
        if isinstance(value, bool):
            raise TypeError("%s must contain integer token IDs, not bool" % name)
        try:
            token_id = operator.index(value)
        except TypeError as exc:
            raise TypeError("%s must contain only integer token IDs" % name) from exc
        result.append(token_id)
    return tuple(result)


def answer_span_extractor(generated_text: str) -> AnswerSpan:
    """Return frozen-regex capture ordinal zero, or fail closed if absent."""

    text = _require_text(generated_text, "generated_text")
    matches = list(ANSWER_REGEX.finditer(text))
    if not matches:
        raise AnswerSpanError("expected at least one answer capture span, found 0")

    selected_match_ordinal = 0
    char_start, char_end = matches[selected_match_ordinal].span(1)
    byte_start = len(text[:char_start].encode("utf-8"))
    byte_end = len(text[:char_end].encode("utf-8"))
    return AnswerSpan(
        char_start=char_start,
        char_end=char_end,
        byte_start=byte_start,
        byte_end=byte_end,
        answer_match_count=len(matches),
        selected_match_ordinal=selected_match_ordinal,
    )


def extract_unique_answer_span(generated_text: str) -> AnswerSpan:
    """Compatibility alias for the evaluator-aligned first-match extractor."""

    return answer_span_extractor(generated_text)


def _decode(
    tokenizer: Any, token_ids: Sequence[int], decode_kwargs: Mapping[str, Any]
) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must provide a callable decode method")
    decoded = decode(list(token_ids), **dict(decode_kwargs))
    if not isinstance(decoded, str):
        raise GeneratedTextAlignmentError("tokenizer.decode must return str")
    return decoded


def _special_ids(tokenizer: Any) -> frozenset:
    values = getattr(tokenizer, "all_special_ids", ())
    if values is None:
        values = ()
    try:
        return frozenset(operator.index(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise GeneratedTextAlignmentError(
            "tokenizer.all_special_ids must be an iterable of integer IDs"
        ) from exc


def _token_byte_spans(
    generated_ids: Sequence[int],
    tokenizer: Any,
    generated_text: str,
    decode_kwargs: Mapping[str, Any],
) -> Tuple[TokenByteSpan, ...]:
    """Return exact spans whose adjacent prefix decodes are both stable.

    Qwen's tokenizer does not guarantee that every intermediate prefix decode
    is a literal prefix of the final decode.  Such transient rewrites are
    irrelevant when they occur away from the selected answer token.  We still
    require exact full-ID/text closure and require the answer start to fall in
    exactly one token interval proven by two adjacent stable prefix decodes.
    """

    full_decoded = _decode(tokenizer, generated_ids, decode_kwargs)
    if full_decoded != generated_text:
        raise GeneratedTextAlignmentError(
            "full generated-ID decode does not exactly round-trip generated_text"
        )

    special_ids = _special_ids(tokenizer)
    stable_prefix_byte_ends = {}
    for stop in range(len(generated_ids) + 1):
        prefix = _decode(tokenizer, generated_ids[:stop], decode_kwargs)
        if generated_text.startswith(prefix):
            stable_prefix_byte_ends[stop] = len(prefix.encode("utf-8"))

    if stable_prefix_byte_ends.get(0) != 0 or stable_prefix_byte_ends.get(
        len(generated_ids)
    ) != len(generated_text.encode("utf-8")):
        raise GeneratedTextAlignmentError(
            "stable decoded prefixes do not close empty/full generated text"
        )

    spans = []
    for index, token_id in enumerate(generated_ids):
        if index not in stable_prefix_byte_ends or index + 1 not in stable_prefix_byte_ends:
            continue
        byte_start = stable_prefix_byte_ends[index]
        byte_end = stable_prefix_byte_ends[index + 1]
        if byte_end < byte_start:
            raise GeneratedTextAlignmentError(
                "adjacent stable decoded-prefix byte offsets are not monotone"
            )
        is_special = int(token_id) in special_ids
        spans.append(
            TokenByteSpan(
                generated_index=index,
                byte_start=byte_start,
                byte_end=byte_end,
                is_special=is_special,
            )
        )
    return tuple(spans)


def validate_generated_text_closure(
    generated_ids: Sequence[int],
    tokenizer: Any,
    generated_text: str,
    *,
    decode_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, ...]:
    """Prove exact full generated-ID/text closure without inspecting payload."""

    text = _require_text(generated_text, "generated_text")
    ids = _token_id_tuple(generated_ids, "generated_ids")
    if not ids:
        raise GeneratedTextAlignmentError("generated_ids must not be empty")
    kwargs = DEFAULT_DECODE_KWARGS if decode_kwargs is None else decode_kwargs
    if not isinstance(kwargs, Mapping):
        raise TypeError("decode_kwargs must be a mapping")
    if _decode(tokenizer, ids, kwargs) != text:
        raise GeneratedTextAlignmentError(
            "full generated-ID decode does not exactly round-trip generated_text"
        )
    return ids


def generated_id_text_aligner(
    generated_ids: Sequence[int],
    tokenizer: Any,
    generated_text: str,
    answer_span: Optional[AnswerSpan] = None,
    *,
    decode_kwargs: Optional[Mapping[str, Any]] = None,
) -> GeneratedAnswerAlignment:
    """Map the selected answer start to its generated token and preceding probe.

    Token intervals are cumulative decoded-prefix UTF-8 byte intervals.  The
    containment test is half-open, so a start at a token boundary belongs to
    the token on the right.
    """

    text = _require_text(generated_text, "generated_text")
    ids = validate_generated_text_closure(
        generated_ids,
        tokenizer,
        text,
        decode_kwargs=decode_kwargs,
    )
    extracted_span = answer_span_extractor(text)
    span = extracted_span if answer_span is None else answer_span
    if not isinstance(span, AnswerSpan):
        raise TypeError("answer_span must be AnswerSpan")
    if span != extracted_span:
        raise GeneratedTextAlignmentError(
            "answer_span does not equal the selected frozen regex capture span"
        )

    expected_byte_start = len(text[: span.char_start].encode("utf-8"))
    expected_byte_end = len(text[: span.char_end].encode("utf-8"))
    if (
        span.char_start < 0
        or span.char_end <= span.char_start
        or span.char_end > len(text)
        or span.byte_start != expected_byte_start
        or span.byte_end != expected_byte_end
    ):
        raise GeneratedTextAlignmentError(
            "answer_span is not an exact span of generated_text"
        )

    kwargs = DEFAULT_DECODE_KWARGS if decode_kwargs is None else decode_kwargs
    if not isinstance(kwargs, Mapping):
        raise TypeError("decode_kwargs must be a mapping")
    token_spans = _token_byte_spans(ids, tokenizer, text, kwargs)
    carriers = [
        token_span
        for token_span in token_spans
        if token_span.byte_start <= span.byte_start < token_span.byte_end
    ]
    if len(carriers) != 1:
        raise GeneratedTextAlignmentError(
            "answer byte start must be carried by exactly one generated token"
        )

    carrier = carriers[0]
    if carrier.generated_index == 0:
        raise NoPrecedingGeneratedTokenError(
            "answer token has no preceding generated probe token"
        )
    if carrier.is_special:
        raise GeneratedTextAlignmentError(
            "answer content cannot be carried by a special token"
        )

    return GeneratedAnswerAlignment(
        answer_span=span,
        answer_first_token_index=carrier.generated_index,
        probe_token_index=carrier.generated_index - 1,
        answer_token_byte_span=carrier.byte_span,
        token_byte_spans=token_spans,
    )


def classify_anchor_eligibility(
    generated_ids: Sequence[int],
    tokenizer: Any,
    generated_text: str,
    *,
    decode_kwargs: Optional[Mapping[str, Any]] = None,
) -> AnchorEligibilityDecision:
    """Classify only the two user-authorized not-expressed conditions.

    A present match with any other alignment failure propagates as an
    engineering error and therefore cannot be absorbed by the mask.
    """

    validate_generated_text_closure(
        generated_ids,
        tokenizer,
        generated_text,
        decode_kwargs=decode_kwargs,
    )
    try:
        span = answer_span_extractor(generated_text)
    except AnswerSpanError:
        return AnchorEligibilityDecision(
            state=ANCHOR_NOT_EXPRESSED,
            answer_span=None,
            alignment=None,
        )
    try:
        alignment = generated_id_text_aligner(
            generated_ids,
            tokenizer,
            generated_text,
            span,
            decode_kwargs=decode_kwargs,
        )
    except NoPrecedingGeneratedTokenError:
        return AnchorEligibilityDecision(
            state=ANCHOR_NOT_EXPRESSED,
            answer_span=span,
            alignment=None,
        )
    return AnchorEligibilityDecision(
        state=ANCHOR_ELIGIBLE,
        answer_span=span,
        alignment=alignment,
    )


def align_generated_ids_to_answer_span(
    generated_ids: Sequence[int],
    tokenizer: Any,
    generated_text: str,
    answer_span: Optional[AnswerSpan] = None,
    *,
    decode_kwargs: Optional[Mapping[str, Any]] = None,
) -> GeneratedAnswerAlignment:
    """Descriptive alias for :func:`generated_id_text_aligner`."""

    return generated_id_text_aligner(
        generated_ids,
        tokenizer,
        generated_text,
        answer_span,
        decode_kwargs=decode_kwargs,
    )


def replay_prefix_ids(
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    answer_first_token_index: int,
    replay_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Build and optionally verify the exact Phase 6 replay prefix.

    The generated portion ends strictly before ``answer_first_token_index``.
    A preceding generated token is mandatory; prompt tokens do not substitute
    for the missing generated probe.
    """

    prompt = _token_id_tuple(prompt_ids, "prompt_ids")
    generated = _token_id_tuple(generated_ids, "generated_ids")
    if not prompt:
        raise ReplayPrefixError("prompt_ids must not be empty")
    if not generated:
        raise ReplayPrefixError("generated_ids must not be empty")
    if isinstance(answer_first_token_index, bool) or not isinstance(
        answer_first_token_index, int
    ):
        raise TypeError("answer_first_token_index must be int")
    if answer_first_token_index <= 0:
        raise ReplayPrefixError(
            "answer token must have a preceding generated probe token"
        )
    if answer_first_token_index >= len(generated):
        raise ReplayPrefixError("answer_first_token_index is out of range")

    expected = prompt + generated[:answer_first_token_index]
    if replay_ids is not None:
        actual = _token_id_tuple(replay_ids, "replay_ids")
        if actual != expected:
            raise ReplayPrefixError(
                "replay IDs do not exactly match prompt plus generated prefix"
            )
    return expected


def build_replay_prefix(
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    answer_first_token_index: int,
    replay_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Descriptive alias for :func:`replay_prefix_ids`."""

    return replay_prefix_ids(
        prompt_ids,
        generated_ids,
        answer_first_token_index,
        replay_ids,
    )


def verify_replay_prefix(
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    answer_first_token_index: int,
    replay_ids: Sequence[int],
) -> Tuple[int, ...]:
    """Require an exact replay-ID match and return the verified prefix."""

    return replay_prefix_ids(
        prompt_ids,
        generated_ids,
        answer_first_token_index,
        replay_ids,
    )


__all__ = [
    "ANCHOR_ELIGIBLE",
    "ANCHOR_NOT_EXPRESSED",
    "ANSWER_REGEX_PATTERN",
    "ANSWER_SPAN_EXTRACTOR_SHA256",
    "ANSWER_SPAN_EXTRACTOR_VERSION",
    "DEFAULT_DECODE_KWARGS",
    "DEFAULT_TEMPLATE_YAML_SHA256",
    "GENERATED_ID_TEXT_ALIGNER_SHA256",
    "GENERATED_ID_TEXT_ALIGNER_VERSION",
    "LM_EVAL_UTILS_SHA256",
    "LM_EVAL_VERSION",
    "MMLU_PRO_YAML_SHA256",
    "REPLAY_PREFIX_SHA256",
    "REPLAY_PREFIX_VERSION",
    "SOURCE_BUNDLE_IDENTITY",
    "AnchorResolutionError",
    "AnchorEligibilityDecision",
    "AnswerSpan",
    "AnswerSpanError",
    "GeneratedAnswerAlignment",
    "GeneratedTextAlignmentError",
    "NoPrecedingGeneratedTokenError",
    "ReplayPrefixError",
    "TokenByteSpan",
    "align_generated_ids_to_answer_span",
    "answer_span_extractor",
    "build_replay_prefix",
    "extract_unique_answer_span",
    "classify_anchor_eligibility",
    "generated_id_text_aligner",
    "replay_prefix_ids",
    "verify_replay_prefix",
    "validate_generated_text_closure",
]
