"""Loop-effect audit entrypoint."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tflt.config import LoopConfig
from tflt.models import resolve_model
from tflt.wrapper import apply_loop_wrapper, _find_layer_owner


DEFAULT_PROMPTS = [
    {
        "id": "mmlu_style_handwritten",
        "kind": "手写 MMLU 风格选择题",
        "text": (
            "The following are multiple choice questions.\n\n"
            "Question: Which gas do plants primarily absorb from the atmosphere during "
            "photosynthesis?\n"
            "A. Oxygen\n"
            "B. Nitrogen\n"
            "C. Carbon dioxide\n"
            "D. Hydrogen\n"
            "Answer:"
        ),
    },
    {
        "id": "short_prompt",
        "kind": "普通短 prompt",
        "text": "Write one concise sentence about why numerical methods need stability.",
    },
]


class AuditCollector:
    """Mutable collector consumed by loop wrappers during an audit run."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.forward_records: List[Dict[str, Any]] = []
        self.tensor_diffs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record(self, event: str, payload: Dict[str, Any]) -> None:
        safe_payload = _jsonable(payload)
        self.events.append({"event": event, "payload": safe_payload})
        self.counts[event] += 1
        wrapper_type = payload.get("wrapper_type")
        if event == "wrapper_forward":
            if wrapper_type:
                self.counts[f"{wrapper_type}_wrapper_forward"] += 1
            if payload.get("bypass"):
                self.counts["bypass_true"] += 1
            else:
                self.counts["bypass_false"] += 1
            self.forward_records.append(safe_payload)
        elif event == "body_call":
            self.counts["operator_body_calls"] += 1
            if wrapper_type:
                self.counts[f"{wrapper_type}_body_calls"] += 1
        elif event == "stash_pass":
            self.counts["stash_passes"] += 1
        elif event == "identity_forward":
            self.counts["identity_layer_forwards"] += 1
        elif event == "bypass":
            self.counts["bypass_events"] += 1
        elif event == "controller_decision":
            self.counts["controller_decisions"] += 1
        elif event == "controller_health":
            self.counts["controller_health_events"] += 1
            self.counts["k_used_total"] += int(payload["k_used"])
            self.counts["declared_operator_body_calls"] += int(payload["operator_body_calls"])

    def record_tensor_diff(self, name: str, before: Any, after: Any) -> None:
        self.tensor_diffs[name].append(diff_stats(before, after))

    def summary(self) -> Dict[str, Any]:
        summary = {
            "counts": dict(self.counts),
            "forward_records": self.forward_records,
            "tensor_diffs": dict(self.tensor_diffs),
        }
        if self.counts.get("controller_health_events", 0):
            summary["controller_health_closed"] = (
                self.counts["operator_body_calls"]
                == self.counts["k_used_total"]
                == self.counts["declared_operator_body_calls"]
            )
        return summary


def add_audit_args(parser: Any) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window", default="12:15")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--iteration-mode", choices=["block", "layer"], default="block")
    parser.add_argument("--strategy", default="damped_euler")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--cache-strategy", choices=["first", "last", "none"], default="last")
    parser.add_argument("--decode-mode", choices=["bypass", "full", "first_n"], default="bypass")
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--prompt", action="append", default=None, help="Add an extra audit prompt.")
    parser.add_argument("--no-use-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)


def cmd_audit_loop_effect(args: Any) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_loop_effect_audit(args)
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "audit_summary.md").write_text(render_summary(report), encoding="utf-8")
    print(str(output_dir / "audit_report.json"))
    print(str(output_dir / "audit_summary.md"))
    return 0


def run_loop_effect_audit(args: Any) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - exercised only on remote eval envs.
        raise RuntimeError("审计需要远端 eval 环境中的 torch 和 transformers。") from exc

    model_info = resolve_model(args.model)
    repo_id = model_info["repo_id"]
    device = _select_device(torch, args.device)
    dtype = _torch_dtype(torch, args.dtype)
    prompts = _audit_prompts(args.prompt)

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    owner, attr = _find_layer_owner(model)
    layers = getattr(owner, attr)
    layer_count = len(layers)
    config_base = _loop_config_dict(args)

    prompt_reports = []
    patch_target: Optional[Dict[str, Any]] = None

    for prompt in prompts:
        collector = AuditCollector()
        config = LoopConfig.from_window_string(
            model_alias=args.model,
            window=args.window,
            k=args.k,
            iteration_mode=args.iteration_mode,
            strategy=args.strategy,
            alpha=args.alpha,
            beta=args.beta,
            cache_strategy=args.cache_strategy,
            decode_mode=args.decode_mode,
            first_n=args.first_n,
            audit_collector=collector,
        )

        inputs = tokenizer(prompt["text"], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        before_classes = _window_classes(layers, config.window)

        with torch.inference_mode():
            baseline_logits = _forward_logits(model, inputs, use_cache=args.use_cache)
            handle = apply_loop_wrapper(model, config)
            patched_layers = getattr(handle.layer_owner, handle.layer_attr)
            after_classes = _window_classes(patched_layers, config.window)
            if patch_target is None:
                patch_target = {
                    "owner_path": _owner_path(model, handle.layer_owner, handle.layer_attr),
                    "layer_attr": handle.layer_attr,
                    "layer_count": layer_count,
                    "window": list(config.window),
                    "classes_before_patch": before_classes,
                    "classes_after_patch": after_classes,
                }
            loop_logits = _forward_logits(model, inputs, use_cache=args.use_cache)
            handle.restore()
            restored_classes = _window_classes(layers, config.window)
            restored_logits = _forward_logits(model, inputs, use_cache=args.use_cache)

        final_diff = diff_stats(baseline_logits, loop_logits)
        restore_diff = diff_stats(baseline_logits, restored_logits)
        choice_delta = choice_logprob_delta(tokenizer, baseline_logits, loop_logits)
        prompt_report = {
            "prompt_id": prompt["id"],
            "prompt_kind_zh": prompt["kind"],
            "prompt_text": prompt["text"],
            "token_count": int(inputs["input_ids"].shape[-1]),
            "patch_classes_before": before_classes,
            "patch_classes_after": after_classes,
            "patch_classes_after_restore": restored_classes,
            "call_stats": collector.summary(),
            "numeric_diff": {
                "final_logits_baseline_vs_loop": final_diff,
                "baseline_before_vs_after_restore": restore_diff,
                "choice_token_logprob_delta": choice_delta,
            },
        }
        prompt_report["decision"] = classify_prompt(prompt_report)
        prompt_reports.append(prompt_report)

        del baseline_logits, loop_logits, restored_logits
        if device == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "report_language": "中文",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audit_scope_zh": "只审计 loop forward 是否真实改变 hidden/logits；未运行 full MMLU、4B 或 ablation。",
        "model": {
            "alias": args.model,
            "repo_id": repo_id,
            "commit_hash": getattr(getattr(model, "config", None), "_commit_hash", None),
            "model_type": getattr(getattr(model, "config", None), "model_type", None),
        },
        "runtime": {
            "device": device,
            "dtype": args.dtype,
            "use_cache": bool(args.use_cache),
            "env": _env_snapshot(),
            "versions": {
                "torch": getattr(torch, "__version__", None),
                "transformers": _module_version("transformers"),
                "tflt": _module_version("tflt"),
            },
        },
        "loop_config": config_base,
        "patch_target": patch_target,
        "prompts": prompt_reports,
    }
    result["overall_decision"] = classify_overall(prompt_reports)
    return result


def _forward_logits(model: Any, inputs: Dict[str, Any], use_cache: bool) -> Any:
    outputs = model(**inputs, use_cache=use_cache)
    return outputs.logits.detach()


def _audit_prompts(extra_prompts: Optional[Iterable[str]]) -> List[Dict[str, str]]:
    prompts = list(DEFAULT_PROMPTS)
    if extra_prompts:
        for idx, text in enumerate(extra_prompts, start=1):
            prompts.append({"id": f"extra_{idx}", "kind": "用户补充 prompt", "text": text})
    return prompts


def _loop_config_dict(args: Any) -> Dict[str, Any]:
    return {
        "window": args.window,
        "k": args.k,
        "iteration_mode": args.iteration_mode,
        "strategy": args.strategy,
        "alpha": args.alpha,
        "beta": args.beta,
        "cache_strategy": args.cache_strategy,
        "decode_mode": args.decode_mode,
        "first_n": args.first_n,
    }


def _select_device(torch_module: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("请求使用 cuda，但当前环境 torch.cuda.is_available() 为 False。")
    return requested


def _torch_dtype(torch_module: Any, dtype: str) -> Any:
    mapping = {
        "auto": "auto",
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    return mapping.get(str(dtype).lower(), dtype)


def _window_classes(layers: Any, window: Tuple[int, int]) -> Dict[str, str]:
    return {str(idx): _class_name(layers[idx]) for idx in range(window[0], window[1] + 1)}


def _class_name(value: Any) -> str:
    return value.__class__.__module__ + "." + value.__class__.__name__


def _owner_path(model: Any, owner: Any, attr: str) -> str:
    candidates = [
        ("<model>", model),
        ("model", getattr(model, "model", None)),
        ("transformer", getattr(model, "transformer", None)),
        ("gpt_neox", getattr(model, "gpt_neox", None)),
        ("decoder", getattr(model, "decoder", None)),
        ("model.decoder", getattr(getattr(model, "model", None), "decoder", None)),
        ("base_model.model", getattr(getattr(model, "base_model", None), "model", None)),
    ]
    for name, candidate in candidates:
        if candidate is owner:
            return f"{name}.{attr}"
    return f"<unknown>.{attr}"


def diff_stats(before: Any, after: Any) -> Dict[str, Any]:
    try:
        import torch

        if hasattr(before, "detach") and hasattr(after, "detach"):
            lhs = before.detach().float()
            rhs = after.detach().float()
            diff = rhs - lhs
            max_abs = float(diff.abs().max().item())
            mean_abs = float(diff.abs().mean().item())
            rms = float(torch.sqrt(torch.mean(diff * diff)).item())
            return {
                "shape": list(lhs.shape),
                "dtype_before": str(getattr(before, "dtype", "")),
                "dtype_after": str(getattr(after, "dtype", "")),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "rms": rms,
                "allclose_rtol_1e-5_atol_1e-6": bool(
                    torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-6)
                ),
                "is_exact_zero": bool(max_abs == 0.0),
            }
    except Exception:
        pass
    try:
        delta = float(after) - float(before)
        return {
            "shape": [],
            "max_abs": abs(delta),
            "mean_abs": abs(delta),
            "rms": abs(delta),
            "allclose_rtol_1e-5_atol_1e-6": bool(math.isclose(float(before), float(after))),
            "is_exact_zero": bool(delta == 0.0),
        }
    except Exception as exc:
        return {"error": str(exc)}


def choice_logprob_delta(tokenizer: Any, baseline_logits: Any, loop_logits: Any) -> Dict[str, Any]:
    try:
        import torch

        base_logprobs = torch.log_softmax(baseline_logits[0, -1].float(), dim=-1)
        loop_logprobs = torch.log_softmax(loop_logits[0, -1].float(), dim=-1)
        result = {}
        for choice in ["A", "B", "C", "D"]:
            token_ids = tokenizer.encode(" " + choice, add_special_tokens=False)
            if not token_ids:
                token_ids = tokenizer.encode(choice, add_special_tokens=False)
            token_id = int(token_ids[0])
            base_value = float(base_logprobs[token_id].item())
            loop_value = float(loop_logprobs[token_id].item())
            base_rank = int((base_logprobs > base_logprobs[token_id]).sum().item() + 1)
            loop_rank = int((loop_logprobs > loop_logprobs[token_id]).sum().item() + 1)
            result[choice] = {
                "token_id": token_id,
                "baseline_logprob": base_value,
                "loop_logprob": loop_value,
                "delta": loop_value - base_value,
                "baseline_rank": base_rank,
                "loop_rank": loop_rank,
            }
        return result
    except Exception as exc:
        return {"error": str(exc)}


def classify_prompt(prompt_report: Dict[str, Any]) -> Dict[str, Any]:
    counts = prompt_report["call_stats"]["counts"]
    tensor_diffs = prompt_report["call_stats"]["tensor_diffs"]
    final_diff = prompt_report["numeric_diff"]["final_logits_baseline_vs_loop"]
    restore_diff = prompt_report["numeric_diff"]["baseline_before_vs_after_restore"]
    block_forwards = int(counts.get("block_wrapper_forward", 0))
    layer_forwards = int(counts.get("layer_wrapper_forward", 0))
    body_calls = int(counts.get("operator_body_calls", 0))
    bypass_true = int(counts.get("bypass_true", 0))
    bypass_false = int(counts.get("bypass_false", 0))
    final_max = float(final_diff.get("max_abs", 0.0))
    restore_ok = bool(restore_diff.get("allclose_rtol_1e-5_atol_1e-6", False))
    g_diffs = tensor_diffs.get("g_minus_x", [])
    g_max = max((float(item.get("max_abs", 0.0)) for item in g_diffs), default=0.0)
    hidden_diffs = tensor_diffs.get("looped_hidden_vs_input", [])
    hidden_max = max((float(item.get("max_abs", 0.0)) for item in hidden_diffs), default=0.0)

    if not restore_ok:
        code = "restore_state_polluted"
        zh = "baseline-after-restore 与 baseline-before 不一致，说明 patch/restore 可能有状态污染。"
    elif block_forwards + layer_forwards == 0:
        code = "wrapper_forward_zero"
        zh = "wrapper forward 调用次数为 0，优先检查 patch target 是否错误。"
    elif bypass_true > 0 and bypass_false == 0:
        code = "wrapper_called_but_all_bypassed"
        zh = "wrapper 被调用，但所有 forward 都触发 decode_mode=bypass，loop body 没有执行。"
    elif body_calls > 0 and g_max <= 1e-12:
        code = "body_called_but_g_minus_x_near_zero"
        zh = "loop body 执行了，但 G(x)-x 近似 0，需检查 block 层是否真实运行或 hidden 提取是否正确。"
    elif hidden_max > 1e-12 and final_max <= 1e-12:
        code = "hidden_changed_but_logits_unchanged"
        zh = "looped hidden 非零变化，但 final logits 近似不变，需检查 stash/identity/replace 逻辑。"
    elif final_max > 1e-12:
        code = "loop_effective_logits_changed"
        zh = "loop 生效，final logits 发生非零变化；若选择 token 排名不变，说明指标不变可能来自决策边界未变。"
    else:
        code = "no_numeric_effect_detected"
        zh = "未检测到 logits 数值变化，但不属于更具体的错误分支，需要进一步查看原始统计。"
    return {
        "code": code,
        "中文结论": zh,
        "restore_allclose": restore_ok,
        "block_wrapper_forward": block_forwards,
        "operator_body_calls": body_calls,
        "bypass_true": bypass_true,
        "bypass_false": bypass_false,
        "max_g_minus_x": g_max,
        "max_looped_hidden_vs_input": hidden_max,
        "max_final_logits_diff": final_max,
    }


def classify_overall(prompt_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    codes = [item["decision"]["code"] for item in prompt_reports]
    if any(code == "restore_state_polluted" for code in codes):
        code = "restore_state_polluted"
        zh = "至少一个 prompt 的 restore 检查失败，优先修复 wrapper restore。"
    elif any(code == "loop_effective_logits_changed" for code in codes):
        code = "loop_effective_logits_changed"
        zh = "至少一个 prompt 的 final logits 被 loop 改变；loop forward 真实生效。"
    elif all(code == "wrapper_called_but_all_bypassed" for code in codes):
        code = "wrapper_called_but_all_bypassed"
        zh = "所有 prompt 都是 wrapper 被调用但 loop body 全 bypass；当前配置下 loop 没有进入主体计算。"
    elif all(code == "wrapper_forward_zero" for code in codes):
        code = "wrapper_forward_zero"
        zh = "所有 prompt 的 wrapper forward 调用为 0，patch target 很可能错误。"
    else:
        code = "mixed_or_no_numeric_effect"
        zh = "审计结果不是单一分支，请逐 prompt 查看 call stats 和 numeric diff。"
    return {"code": code, "中文结论": zh, "prompt_codes": codes}


def render_summary(report: Dict[str, Any]) -> str:
    lines = [
        "# Loop 生效性审计摘要",
        "",
        "## 结论",
        "",
        f"- 总体分类：`{report['overall_decision']['code']}`",
        f"- 中文结论：{report['overall_decision']['中文结论']}",
        "- 本审计只运行固定 prompt forward，没有启动 full MMLU、4B、RK/window/cache ablation。",
        "",
        "## 模型与配置",
        "",
        f"- 模型：`{report['model']['repo_id']}`",
        f"- 模型 commit：`{report['model'].get('commit_hash')}`",
        f"- dtype：`{report['runtime']['dtype']}`",
        f"- device：`{report['runtime']['device']}`",
        f"- use_cache：`{report['runtime']['use_cache']}`",
        f"- loop 配置：`{json.dumps(report['loop_config'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Patch Target",
        "",
    ]
    patch = report.get("patch_target") or {}
    lines.extend(
        [
            f"- owner path：`{patch.get('owner_path')}`",
            f"- layer attr：`{patch.get('layer_attr')}`",
            f"- layer count：`{patch.get('layer_count')}`",
            f"- window：`{patch.get('window')}`",
            f"- patch 前 class：`{patch.get('classes_before_patch')}`",
            f"- patch 后 class：`{patch.get('classes_after_patch')}`",
            "",
            "## Prompt 结果",
            "",
            "| prompt | 分类 | block forward | body calls | bypass true/false | max G(x)-x | max hidden diff | max logits diff | restore allclose |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["prompts"]:
        d = item["decision"]
        lines.append(
            "| {pid} | {code} | {bf} | {body} | {bt}/{bfal} | {gx:.6g} | {hd:.6g} | {ld:.6g} | {restore} |".format(
                pid=item["prompt_id"],
                code=d["code"],
                bf=d["block_wrapper_forward"],
                body=d["operator_body_calls"],
                bt=d["bypass_true"],
                bfal=d["bypass_false"],
                gx=d["max_g_minus_x"],
                hd=d["max_looped_hidden_vs_input"],
                ld=d["max_final_logits_diff"],
                restore=d["restore_allclose"],
            )
        )
    lines.extend(["", "## 说明", ""])
    for item in report["prompts"]:
        lines.append(f"- `{item['prompt_id']}`：{item['decision']['中文结论']}")
    return "\n".join(lines) + "\n"


def _env_snapshot() -> Dict[str, str]:
    keys = [
        "HF_ENDPOINT",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_DISABLE_XET",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
