"""Pure-Python Phase 6 final-answer anchor resolution.

The extractor deliberately differs from lm-eval's outcome filter: lm-eval may
apply ``take_first`` after extraction, while Phase 6 fails closed unless the
capture group has exactly one span.  The aligner uses only generated token IDs,
the tokenizer's actual prefix decodes, and the generated text supplied by the
generation path.
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

ANSWER_SPAN_EXTRACTOR_VERSION = "loopscope-phase6-answer-span-v1"
GENERATED_ID_TEXT_ALIGNER_VERSION = "loopscope-phase6-generated-id-aligner-v1"
REPLAY_PREFIX_VERSION = "loopscope-phase6-replay-prefix-v1"

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
    match_policy="all_capture_group_1_spans_exactly_one",
)
GENERATED_ID_TEXT_ALIGNER_SHA256 = _identity_hash(
    "generated_id_text_aligner",
    GENERATED_ID_TEXT_ALIGNER_VERSION,
    span_unit="utf8_bytes",
    boundary_policy="half_open_boundary_belongs_right",
    decode_kwargs=dict(DEFAULT_DECODE_KWARGS),
    full_text_round_trip="exact",
    special_token_content="forbidden",
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
    """The frozen answer regex did not produce exactly one capture span."""


class GeneratedTextAlignmentError(AnchorResolutionError):
    """Generated IDs and text could not be aligned exactly."""


class ReplayPrefixError(AnchorResolutionError):
    """The replay prefix was invalid or did not match the frozen IDs."""


@dataclass(frozen=True)
class AnswerSpan:
    """The unique capture-group-1 span in character and UTF-8 byte units."""

    char_start: int
    char_end: int
    byte_start: int
    byte_end: int

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
    """Resolved generated-token location for a unique answer span."""

    answer_span: AnswerSpan
    answer_first_token_index: int
    probe_token_index: int
    answer_token_byte_span: Tuple[int, int]
    token_byte_spans: Tuple[TokenByteSpan, ...]
    extractor_sha256: str = ANSWER_SPAN_EXTRACTOR_SHA256
    aligner_sha256: str = GENERATED_ID_TEXT_ALIGNER_SHA256


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
    """Return the sole frozen-regex capture span, or fail closed.

    All capture-group-1 spans are collected before deciding.  Repeated matches
    are ambiguous even when they capture the same letter.
    """

    text = _require_text(generated_text, "generated_text")
    spans = [match.span(1) for match in ANSWER_REGEX.finditer(text)]
    if len(spans) != 1:
        raise AnswerSpanError(
            "expected exactly one answer capture span, found %d" % len(spans)
        )

    char_start, char_end = spans[0]
    byte_start = len(text[:char_start].encode("utf-8"))
    byte_end = len(text[:char_end].encode("utf-8"))
    return AnswerSpan(
        char_start=char_start,
        char_end=char_end,
        byte_start=byte_start,
        byte_end=byte_end,
    )


def extract_unique_answer_span(generated_text: str) -> AnswerSpan:
    """Descriptive alias for :func:`answer_span_extractor`."""

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
    full_decoded = _decode(tokenizer, generated_ids, decode_kwargs)
    if full_decoded != generated_text:
        raise GeneratedTextAlignmentError(
            "full generated-ID decode does not exactly round-trip generated_text"
        )

    special_ids = _special_ids(tokenizer)
    prefixes = []
    for stop in range(len(generated_ids) + 1):
        prefix = _decode(tokenizer, generated_ids[:stop], decode_kwargs)
        if not generated_text.startswith(prefix):
            raise GeneratedTextAlignmentError(
                "decoded prefix %d is not an exact generated_text prefix" % stop
            )
        prefixes.append(prefix)

    if prefixes[0] != "" or prefixes[-1] != generated_text:
        raise GeneratedTextAlignmentError(
            "decoded prefixes do not cover generated_text exactly"
        )

    spans = []
    previous_end = 0
    for index, token_id in enumerate(generated_ids):
        byte_start = len(prefixes[index].encode("utf-8"))
        byte_end = len(prefixes[index + 1].encode("utf-8"))
        if byte_start != previous_end or byte_end < byte_start:
            raise GeneratedTextAlignmentError(
                "decoded-prefix UTF-8 byte spans are not cumulative"
            )
        is_special = int(token_id) in special_ids
        if is_special and byte_end != byte_start:
            raise GeneratedTextAlignmentError(
                "special token at generated index %d contributes visible content"
                % index
            )
        spans.append(
            TokenByteSpan(
                generated_index=index,
                byte_start=byte_start,
                byte_end=byte_end,
                is_special=is_special,
            )
        )
        previous_end = byte_end

    if previous_end != len(generated_text.encode("utf-8")):
        raise GeneratedTextAlignmentError(
            "decoded-prefix spans do not cover generated_text UTF-8 bytes"
        )
    return tuple(spans)


def generated_id_text_aligner(
    generated_ids: Sequence[int],
    tokenizer: Any,
    generated_text: str,
    answer_span: Optional[AnswerSpan] = None,
    *,
    decode_kwargs: Optional[Mapping[str, Any]] = None,
) -> GeneratedAnswerAlignment:
    """Map the unique answer start to its generated token and preceding probe.

    Token intervals are cumulative decoded-prefix UTF-8 byte intervals.  The
    containment test is half-open, so a start at a token boundary belongs to
    the token on the right.
    """

    text = _require_text(generated_text, "generated_text")
    ids = _token_id_tuple(generated_ids, "generated_ids")
    if not ids:
        raise GeneratedTextAlignmentError("generated_ids must not be empty")
    extracted_span = answer_span_extractor(text)
    span = extracted_span if answer_span is None else answer_span
    if not isinstance(span, AnswerSpan):
        raise TypeError("answer_span must be AnswerSpan")
    if span != extracted_span:
        raise GeneratedTextAlignmentError(
            "answer_span does not equal the frozen regex capture span"
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
    if carrier.is_special:
        raise GeneratedTextAlignmentError(
            "answer content cannot be carried by a special token"
        )
    if carrier.generated_index == 0:
        raise GeneratedTextAlignmentError(
            "answer token has no preceding generated probe token"
        )

    return GeneratedAnswerAlignment(
        answer_span=span,
        answer_first_token_index=carrier.generated_index,
        probe_token_index=carrier.generated_index - 1,
        answer_token_byte_span=carrier.byte_span,
        token_byte_spans=token_spans,
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
    "AnswerSpan",
    "AnswerSpanError",
    "GeneratedAnswerAlignment",
    "GeneratedTextAlignmentError",
    "ReplayPrefixError",
    "TokenByteSpan",
    "align_generated_ids_to_answer_span",
    "answer_span_extractor",
    "build_replay_prefix",
    "extract_unique_answer_span",
    "generated_id_text_aligner",
    "replay_prefix_ids",
    "verify_replay_prefix",
]
