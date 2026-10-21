#!/usr/bin/env python3
"""Compare the mounted model with public fixed-revision bytes before any forward.

This cloud-only check streams public files without persisting a second model.
It deliberately uses neither credentials nor content digests.
"""
import argparse
import json
import signal
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPOSITORY = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
FILES = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "generation_config.json",
)
CHUNK_BYTES = 8 * 1024 * 1024


def event(state, stage, **fields):
    state.update(stage=stage, **fields)
    print(json.dumps({"event": "model_source", **state}), flush=True)


def network_error(exc):
    details = {"error_type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        details["http_status"] = exc.code
    elif isinstance(exc, urllib.error.URLError):
        details.update(reason_type=type(exc.reason).__name__,
                       reason_errno=getattr(exc.reason, "errno", None))
        if getattr(exc.reason, "verify_message", None):
            details["certificate_error"] = exc.reason.verify_message
    return details


def compare_file(model_dir, filename, deadline, endpoint="https://huggingface.co", state=None):
    state = {} if state is None else state
    event(state, "local_stat", file=filename, bytes_compared=0, endpoint=endpoint)
    local_path = model_dir / filename
    before = local_path.stat()
    if not local_path.is_file():
        raise ValueError("required model path is not a regular file")
    url = f"{endpoint}/{REPOSITORY}/resolve/{REVISION}/{filename}"
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    compared = 0
    event(state, "remote_connect", local_bytes=before.st_size)
    with local_path.open("rb") as local, urllib.request.urlopen(
        request, timeout=min(20, max(0.001, deadline - time.monotonic()))
    ) as remote:
        event(state, "remote_headers", http_status=remote.status)
        if remote.status != 200:
            raise ValueError(f"unexpected HTTP status {remote.status}")
        if remote.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("unexpected encoded response")
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("whole-operation deadline exceeded")
            state["stage"] = "remote_read"
            block = remote.read(CHUNK_BYTES)
            if not block:
                if local.read(1):
                    raise ValueError(f"remote ended before local at byte {compared}")
                break
            state["stage"] = "mounted_read"
            mounted = local.read(len(block))
            if mounted != block:
                raise ValueError(f"byte mismatch in block starting at {compared}")
            compared += len(block)
            state["bytes_compared"] = compared
            if compared % (128 * 1024 * 1024) == 0:
                event(state, "stream_progress")
        length = remote.headers.get("Content-Length")
        if length is not None and int(length) != compared:
            raise ValueError("response length does not equal bytes consumed")
    after = local_path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("local model file changed during comparison")
    if compared != before.st_size:
        raise ValueError("compared length does not equal local size")
    return {"file": filename, "bytes_compared": compared, "equal": True,
            "reference_url": url}


def probe(model_dir, deadline):
    """Bounded cloud-only endpoint diagnosis; never counts as source admission."""
    checks = []
    filename = FILES[0]
    for endpoint in ("https://hf-mirror.com", "https://huggingface.co"):
        state = {"endpoint": endpoint, "file": filename}
        began = time.monotonic()
        try:
            event(state, "local_prefix_read")
            with (model_dir / filename).open("rb") as mounted:
                prefix = mounted.read(65536)
            state["local_prefix_seconds"] = round(time.monotonic()-began, 3)
            url = f"{endpoint}/{REPOSITORY}/resolve/{REVISION}/{filename}"
            request = urllib.request.Request(url, headers={"Range": "bytes=0-65535", "Accept-Encoding": "identity"})
            event(state, "remote_connect")
            with urllib.request.urlopen(request, timeout=min(12, max(.001, deadline-time.monotonic()))) as response:
                state["http_status"] = response.status
                event(state, "remote_prefix_read")
                block = response.read(65536)
            state.update(status="PREFIX_ONLY_EQUAL" if block == prefix else "PREFIX_DIFFERENT",
                         bytes_received=len(block))
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise
            state.update(status="ACCESS_FAILED", **network_error(exc))
        state["elapsed_seconds"] = round(time.monotonic()-began, 3)
        event(state, "probe_result")
        checks.append(state)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"),
                        default="https://hf-mirror.com")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.timeout_seconds < float("inf"):
        parser.error("--timeout-seconds must be finite and positive")
    # Reserve a fresh evidence file before accessing the model or network.
    with args.output.open("x", encoding="utf-8") as output:
        started = time.monotonic()
        report = {"status": "FAIL", "repository": REPOSITORY, "revision": REVISION,
                  "model_dir": str(args.model_dir.resolve()), "files": [],
                  "comparison": "direct_full_stream_bytes", "timeout_seconds": args.timeout_seconds}
        current_file = None
        state = {}
        previous_handler = signal.getsignal(signal.SIGALRM)

        def expired(_signum, _frame):
            raise TimeoutError("whole-operation deadline exceeded")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        try:
            if args.probe_only:
                report["endpoint_probes"] = probe(args.model_dir, started + args.timeout_seconds)
                report["status"] = "NETWORK_PROBE_COMPLETE_NOT_SOURCE_ADMITTED"
            else:
                for current_file in FILES:
                    report["files"].append(compare_file(
                        args.model_dir, current_file, started + args.timeout_seconds, args.endpoint, state))
                    print(f"equal {current_file}", flush=True)
                report["status"] = "PASS"
        except Exception as exc:
            # Do not record signed redirect URLs, response bodies, or credentials.
            report["failed_file"] = current_file
            report["error_type"] = type(exc).__name__
            report["last_progress"] = state
            report.update(network_error(exc))
            report["error"] = str(exc)[:300] if isinstance(exc, (ValueError, TimeoutError)) else "file or public reference access failed"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            json.dump(report, output, indent=2)
            output.write("\n")
            output.flush()
    return 0 if report["status"] in ("PASS", "NETWORK_PROBE_COMPLETE_NOT_SOURCE_ADMITTED") else 1


if __name__ == "__main__":
    sys.exit(main())
