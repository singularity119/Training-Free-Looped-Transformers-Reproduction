"""Small architecture adapters and raw-boundary helpers for Phase 7.

No ML dependency is imported at module import time.  The functions accept
ordinary module-like objects, which keeps the local fake-module tests useful on
the dependency-free development machine while preserving the real HF path for
the later Gate B smoke.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase7_schema import MODEL_SPEC_BY_KEY, Phase7ContractError


class Phase7RuntimeError(Phase7ContractError):
    """Runtime path or boundary-capture failure."""


@dataclass(frozen=True)
class ArchitectureAdapter:
    model_key: str
    architecture_family: str
    decoder_container_path: str
    decoder_blocks_path: str
    embedding_path: str
    final_norm_path: str
    lm_head_path: str

    @property
    def path_contract(self) -> Dict[str, str]:
        return {
            "decoder_container": self.decoder_container_path,
            "decoder_blocks": self.decoder_blocks_path,
            "embedding": self.embedding_path,
            "final_norm": self.final_norm_path,
            "lm_head": self.lm_head_path,
        }


def get_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split(".") if path else ():
        if not hasattr(value, part):
            raise Phase7RuntimeError("module path is missing: %s" % path)
        value = getattr(value, part)
    return value


def _find_path(root: Any, candidates: Iterable[str], label: str) -> Tuple[str, Any]:
    for path in candidates:
        try:
            return path, get_path(root, path)
        except Phase7RuntimeError:
            continue
    raise Phase7RuntimeError("could not locate %s" % label)


def _family_from_config(model: Any) -> str:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = tuple(str(value).lower() for value in (getattr(config, "architectures", None) or ()))
    joined = " ".join((model_type,) + architectures)
    if "qwen2" in joined:
        return "qwen2"
    if "llama" in joined:
        return "llama"
    if "gemma2" in joined:
        return "gemma2"
    if "gemma" in joined:
        return "gemma2"
    raise Phase7RuntimeError("unsupported Phase 7 model architecture: %s" % joined.strip())


def resolve_architecture(model: Any, *, model_key: Optional[str] = None) -> ArchitectureAdapter:
    """Resolve the three frozen dense causal-LM layouts without model forward."""

    family = _family_from_config(model)
    if model_key is None:
        family_to_key = {"qwen2": "qwen25_3b", "llama": "llama32_3b", "gemma2": "gemma2_2b"}
        model_key = family_to_key[family]
    if model_key not in MODEL_SPEC_BY_KEY:
        raise Phase7RuntimeError("unknown Phase 7 model key: %s" % model_key)
    expected_family = MODEL_SPEC_BY_KEY[model_key]["architecture_family"]
    if family != expected_family:
        raise Phase7RuntimeError("model architecture family does not match model_key")

    container_path, container = _find_path(model, ("model", "transformer"), "decoder container")
    blocks_path, blocks = _find_path(
        model,
        (
            container_path + ".layers",
            container_path + ".h",
            container_path + ".blocks",
            container_path + ".decoder.layers",
        ),
        "decoder blocks",
    )
    layer_count = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    if layer_count < 1 or not hasattr(blocks, "__len__") or len(blocks) != layer_count:
        raise Phase7RuntimeError("decoder block count does not close to config.num_hidden_layers")
    embedding_path, _embedding = _find_path(
        model,
        (
            container_path + ".embed_tokens",
            container_path + ".wte",
            container_path + ".word_embeddings",
        ),
        "input embedding",
    )
    final_norm_path, _final_norm = _find_path(
        model,
        (
            container_path + ".norm",
            container_path + ".final_layernorm",
            container_path + ".final_layer_norm",
            container_path + ".ln_f",
        ),
        "final norm",
    )
    lm_head_path = "lm_head"
    try:
        get_path(model, lm_head_path)
    except Phase7RuntimeError:
        getter = getattr(model, "get_output_embeddings", None)
        if not callable(getter) or getter() is None:
            raise Phase7RuntimeError("could not locate lm_head/output embeddings")
        # The actual object is valid, but the persisted path must remain a
        # readable path.  A custom output embedding is represented explicitly.
        lm_head_path = "<get_output_embeddings>"
    return ArchitectureAdapter(
        model_key=model_key,
        architecture_family=family,
        decoder_container_path=container_path,
        decoder_blocks_path=blocks_path,
        embedding_path=embedding_path,
        final_norm_path=final_norm_path,
        lm_head_path=lm_head_path,
    )


def resolve_lm_head(model: Any, adapter: ArchitectureAdapter) -> Any:
    if adapter.lm_head_path == "<get_output_embeddings>":
        getter = getattr(model, "get_output_embeddings", None)
        head = getter() if callable(getter) else None
        if head is None:
            raise Phase7RuntimeError("get_output_embeddings returned no module")
        return head
    return get_path(model, adapter.lm_head_path)


def hidden_states_from_output(outputs: Any) -> Tuple[Any, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, Mapping):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is None:
        raise Phase7RuntimeError("model output did not expose hidden_states")
    values = tuple(hidden_states)
    if not values:
        raise Phase7RuntimeError("model output hidden_states is empty")
    return values


def assemble_raw_boundaries(
    hidden_states: Sequence[Any], captured_raw_final: Any, layer_count: int, pre_hook_count: int
) -> Tuple[Any, ...]:
    """Return raw ``B0...BL`` and replace any post-norm endpoint with pre-hook data."""

    if pre_hook_count != 1:
        raise Phase7RuntimeError("FinalNorm pre-hook count must equal one")
    if captured_raw_final is None:
        raise Phase7RuntimeError("FinalNorm pre-hook did not capture raw B_L")
    values = tuple(hidden_states)
    if len(values) == layer_count + 2:
        # HF decoder models commonly expose raw B0...BL plus a final-normalized
        # copy.  The last item is not a raw boundary and is discarded.
        values = values[:-1]
    if len(values) != layer_count + 1:
        raise Phase7RuntimeError("hidden_states do not contain L+1 raw boundaries")
    return tuple(values[:-1]) + (captured_raw_final,)


def select_position_vector(state: Any, position: int) -> Any:
    """Select one position from a [B,S,H] or accept a [B,H] tensor."""

    ndim = getattr(state, "ndim", None)
    shape = tuple(getattr(state, "shape", ()))
    if ndim == 3 and len(shape) == 3 and shape[0] == 1 and shape[1] >= 1:
        if isinstance(position, bool) or position < 0 or position >= shape[1]:
            raise Phase7RuntimeError("probe position is outside the hidden-state sequence")
        return state[:, position, :]
    if ndim == 2 and len(shape) == 2 and shape[0] == 1:
        if position not in (0, -1):
            raise Phase7RuntimeError("a [B,H] boundary has no nonzero sequence position")
        return state
    raise Phase7RuntimeError("boundary tensor is not [1,S,H] or [1,H]")


def select_probe_vector(state: Any) -> Any:
    """Select the final sequence position from a [B,S,H] or [B,H] tensor."""

    shape = tuple(getattr(state, "shape", ()))
    position = shape[1] - 1 if len(shape) == 3 else 0
    return select_position_vector(state, position)


def select_probe_vectors(raw_boundaries: Sequence[Any]) -> Tuple[Any, ...]:
    values = tuple(select_probe_vector(state) for state in raw_boundaries)
    if not values:
        raise Phase7RuntimeError("no raw boundaries were captured")
    widths = {tuple(getattr(value, "shape", ())) for value in values}
    if len(widths) != 1:
        raise Phase7RuntimeError("raw boundary probe vector shapes differ")
    return values


def native_final_hidden(outputs: Any) -> Any:
    value = getattr(outputs, "last_hidden_state", None)
    if value is None and isinstance(outputs, Mapping):
        value = outputs.get("last_hidden_state")
    if value is None:
        raise Phase7RuntimeError("model output did not expose last_hidden_state")
    return select_probe_vector(value)


def register_raw_final_norm_hook(final_norm: Any) -> Tuple[Dict[str, Any], Any]:
    """Register temporary pre/forward hooks for one native FinalNorm call.

    The pre-hook captures raw ``B_L``.  The forward hook captures the output of
    that same native call, allowing the producer to check endpoint closure
    without invoking FinalNorm a second time on ``B_L``.
    """

    register = getattr(final_norm, "register_forward_pre_hook", None)
    if not callable(register):
        raise Phase7RuntimeError("final norm lacks register_forward_pre_hook")
    capture: Dict[str, Any] = {"count": 0, "raw": None}

    def hook(_module: Any, args: Tuple[Any, ...]) -> None:
        capture["count"] += 1
        if not args:
            raise Phase7RuntimeError("FinalNorm pre-hook received no input")
        capture["raw"] = args[0]

    handles = [register(hook)]
    register_output = getattr(final_norm, "register_forward_hook", None)
    if not callable(register_output):
        raise Phase7RuntimeError("final norm lacks register_forward_hook for closure")

    def output_hook(_module: Any, _args: Tuple[Any, ...], output: Any) -> None:
        capture["normalized"] = output

    handles.append(register_output(output_hook))

    class _HandleGroup:
        def remove(self) -> None:
            for item in handles:
                remove = getattr(item, "remove", None)
                if callable(remove):
                    remove()

    return capture, _HandleGroup()


def module_path(root: Any, target: Any) -> str:
    named_modules = getattr(root, "named_modules", None)
    if callable(named_modules):
        for name, module in named_modules():
            if module is target:
                return name or "<root>"
    raise Phase7RuntimeError("target module is not registered under model")


def assert_no_loop_wrapper(model: Any) -> None:
    """Fail closed if the loaded model has the project loop wrapper attached."""

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise Phase7RuntimeError("model lacks named_modules for zero-loop check")
    wrapped = []
    for path, module in named_modules():
        module_type = type(module)
        module_name = str(getattr(module_type, "__name__", ""))
        module_origin = str(getattr(module_type, "__module__", ""))
        if module_name in {"LoopedTransformer", "LoopWrapper", "LoopModel"} or module_origin.startswith("tflt.wrapper"):
            wrapped.append(path or "<root>")
    if wrapped:
        raise Phase7RuntimeError("loop wrapper is attached to model: %s" % ",".join(sorted(wrapped)))


__all__ = [
    "ArchitectureAdapter",
    "Phase7RuntimeError",
    "assemble_raw_boundaries",
    "assert_no_loop_wrapper",
    "get_path",
    "hidden_states_from_output",
    "module_path",
    "native_final_hidden",
    "register_raw_final_norm_hook",
    "resolve_architecture",
    "resolve_lm_head",
    "select_probe_vector",
    "select_position_vector",
    "select_probe_vectors",
]
