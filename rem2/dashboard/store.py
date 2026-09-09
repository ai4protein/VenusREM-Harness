"""Filesystem store for dashboard runs."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_root() -> Path:
    env = (os.environ.get("REM2_DASHBOARD_HOME") or "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "rem2" / "dashboard"


def safe_stem(name: str, fallback: str = "protein") -> str:
    raw = _SAFE.sub("_", (name or "").strip()).strip("._")
    return (raw[:80] or fallback)


class RunStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or default_root())
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def job_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "job.json"

    def log_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "log.txt"

    def inputs_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id) / "inputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def result_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id) / "result"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def write_job(self, job: dict[str, Any]) -> dict[str, Any]:
        run_id = job["id"]
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        job["updated_at"] = utc_now()
        path = self.job_path(run_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        return job

    def load_job(self, run_id: str) -> Optional[dict[str, Any]]:
        path = self.job_path(run_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        if not self.runs_dir.is_dir():
            return jobs
        for child in self.runs_dir.iterdir():
            if not child.is_dir():
                continue
            job = self.load_job(child.name)
            if job:
                jobs.append(job)
        jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return jobs

    def append_log(self, run_id: str, text: str) -> None:
        path = self.log_path(run_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
