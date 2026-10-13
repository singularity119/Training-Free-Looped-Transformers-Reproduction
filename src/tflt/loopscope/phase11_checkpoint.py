"""Single-writer, question-boundary checkpoints; no model or token-level state."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import uuid


class RecordStore:
    """Rebuild progress from validated, atomic record/telemetry bundles.

    The caller owns one run root and supplies a validator that raises for any
    scientific identity, cell, or mechanism mismatch. Those exceptions propagate.
    Only incomplete serialization is quarantined and made pending again.
    """

    def __init__(self, root, indices, validator):
        self.root = Path(root)
        self.indices = list(indices)
        if (not self.indices or len(set(self.indices)) != len(self.indices)
                or any(type(i) is not int or i < 0 for i in self.indices)):
            raise ValueError("checkpoint indices must be nonempty, unique, nonnegative integers")
        if not callable(validator):
            raise TypeError("checkpoint validator must be callable")
        self.validator = validator
        self.complete = self.root / "complete"
        self.quarantine = self.root / "quarantine"
        self.complete.mkdir(parents=True, exist_ok=True)
        self.quarantine.mkdir(exist_ok=True)
        self._done = {}
        self._in_progress = set()
        self.quarantined = []
        self.scan()

    @property
    def done(self):
        return dict(self._done)

    @property
    def pending(self):
        return [i for i in self.indices if i not in self._done]

    def _path(self, index):
        if index not in self.indices or type(index) is not int:
            raise ValueError("checkpoint index outside declared membership")
        return self.complete / f"{index:06d}.json"

    def _atomic_json(self, path, value):
        temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        # On any failure, retain the temporary file for the next scan to isolate.
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _isolate(self, path, reason, exception=None):
        destination = self.quarantine / f"{path.name}.{uuid.uuid4().hex}"
        os.replace(path, destination)
        event = {"source_path": str(path), "preserved_path": str(destination), "reason": reason,
                 "exception_type": None if exception is None else type(exception).__name__,
                 "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        # The original half-written bytes remain untouched beside their reason.
        self._atomic_json(self.quarantine / f"{destination.name}.reason.json", event)
        self.quarantined.append(event)

    def _read_complete(self, index, path):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._isolate(path, "incomplete or invalid JSON serialization: " + str(exc), exc)
            return None
        if not isinstance(bundle, dict):
            raise ValueError("complete bundle must be a record/telemetry object")
        if "record" not in bundle or "telemetry" not in bundle:
            self._isolate(path, "incomplete bundle missing record or telemetry")
            return None
        if not isinstance(bundle["record"], dict) or not isinstance(bundle["telemetry"], dict):
            raise ValueError("complete record and telemetry must be objects")
        self.validator(index, bundle["record"], bundle["telemetry"])
        return bundle

    def _save_state(self):
        self._atomic_json(self.root / "checkpoint.json", {
            "schema": "loopscope.phase11.question_checkpoint.v1", "indices": self.indices,
            "pending": [i for i in self.indices if i not in self._done and i not in self._in_progress],
            "in_progress": [i for i in self.indices if i in self._in_progress],
            "done": [i for i in self.indices if i in self._done],
            "state_source": "validated complete bundles", "quarantine_dir": str(self.quarantine),
            "quarantined_this_process": self.quarantined})

    def scan(self):
        """Ignore prior checkpoint state and validate the on-disk complete set."""
        for directory in (self.root, self.complete):
            for path in sorted(directory.glob("*.tmp")):
                self._isolate(path, "uncommitted temporary file")
        recovered = {}
        for path in sorted(self.complete.glob("*.json")):
            try:
                index = int(path.stem)
            except ValueError as exc:
                raise ValueError("complete bundle filename is outside declared membership") from exc
            if path != self._path(index):
                raise ValueError("complete bundle filename differs from canonical index path")
            bundle = self._read_complete(index, path)
            if bundle is not None:
                recovered[index] = bundle
        self._done = {i: recovered[i] for i in self.indices if i in recovered}
        self._in_progress = set()
        self._save_state()
        return self.done

    def mark_in_progress(self, index):
        self._path(index)
        if index in self._done:
            raise FileExistsError("validated successful checkpoint must not be recomputed")
        self._in_progress.add(index)
        self._save_state()

    def commit(self, index, record, telemetry):
        """Atomically publish one full successful question before updating state."""
        path = self._path(index)
        if index in self._done:
            raise FileExistsError("validated successful checkpoint must never be overwritten")
        if path.exists():
            existing = self._read_complete(index, path)
            if existing is not None:
                self._done[index] = existing
                self._in_progress.discard(index)
                self._save_state()
                raise FileExistsError("validated successful checkpoint already exists")
        if not isinstance(record, dict) or not isinstance(telemetry, dict):
            raise ValueError("record and telemetry must be objects")
        self.validator(index, record, telemetry)
        bundle = {"record": record, "telemetry": telemetry}
        self._atomic_json(path, bundle)
        self._done[index] = bundle
        self._in_progress.discard(index)
        self._save_state()

    def _materialize(self, path, rows):
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                existing = [json.loads(line) for line in handle if line.strip()]
            if existing != rows:
                raise ValueError("existing final output differs from complete checkpoint records")
            return
        temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        with temp.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def finalize_records(self):
        """Validate full closure again and materialize ordered legacy JSONL files."""
        self.scan()
        if self.pending:
            raise ValueError("cannot finalize an incomplete checkpoint population")
        records = self.root / "records.jsonl"
        telemetry = self.root / "telemetry.jsonl"
        self._materialize(records, [self._done[i]["record"] for i in self.indices])
        self._materialize(telemetry, [self._done[i]["telemetry"] for i in self.indices])
        return records, telemetry
