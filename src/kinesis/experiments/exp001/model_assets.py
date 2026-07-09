from __future__ import annotations

from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen

DEFAULT_POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def ensure_pose_landmarker_model(
    model_path: Path,
    *,
    model_url: str = DEFAULT_POSE_LANDMARKER_MODEL_URL,
    timeout_seconds: float = 120.0,
) -> Path:
    """Return a local Pose Landmarker model path, downloading it when needed."""
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    if model_path.exists() and not model_path.is_file():
        raise FileExistsError(f"Model path exists but is not a file: {model_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_name(f"{model_path.name}.download")
    temporary_path.unlink(missing_ok=True)

    try:
        _download_file(
            url=model_url,
            destination=temporary_path,
            timeout_seconds=timeout_seconds,
        )
        temporary_path.replace(model_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return model_path


def _download_file(
    *,
    url: str,
    destination: Path,
    timeout_seconds: float,
) -> None:
    request = Request(url, headers={"User-Agent": "kinesis-movement-lab/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        with destination.open("wb") as output_file:
            copyfileobj(response, output_file)
