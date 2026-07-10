from __future__ import annotations

import os
from pathlib import Path
from tempfile import gettempdir


def runtime_root(project_root: Path) -> Path:
    override = os.environ.get("KINESIS_RUNTIME_DIR")
    if override:
        return Path(override)
    if project_root.as_posix().startswith("/mount/src/"):
        return Path(gettempdir()) / "kinesis"
    return project_root


def resolve_runtime_model_path(project_root: Path, model_path: Path) -> Path:
    if model_path.is_absolute():
        return model_path

    root = runtime_root(project_root)
    if root != project_root:
        return root / model_path
    return project_root / model_path


def run_output_dir(project_root: Path, experiment_id: str, run_id: str) -> Path:
    return runtime_root(project_root) / "outputs" / experiment_id / run_id
