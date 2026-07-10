from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from kinesis.ui.runtime_paths import resolve_runtime_model_path, runtime_root


def test_kinesis_logo_assets_exist() -> None:
    assert Path("assets/kinesis-icon.png").is_file()
    assert Path("assets/kinesis-wordmark.png").is_file()


def test_streamlit_runtime_dir_override_keeps_generated_files_out_of_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KINESIS_RUNTIME_DIR", str(tmp_path))
    project_root = Path("/mount/src/kinesis")

    assert runtime_root(project_root) == tmp_path
    assert (
        resolve_runtime_model_path(project_root, Path("models/pose.task"))
        == tmp_path / "models" / "pose.task"
    )


def test_streamlit_app_renders_exp001_and_exp002_controls() -> None:
    app = AppTest.from_file("apps/exp001_streamlit.py")

    app.run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Kinesis Movement Lab"
    assert "EXP-001: Dance Pose Estimation" in [header.value for header in app.header]
    assert "EXP-002: Reference Movement Matching" in [header.value for header in app.header]
    assert "Dance video" in [uploader.label for uploader in app.file_uploader]
    assert "Reference video" in [uploader.label for uploader in app.file_uploader]
    assert "Practice video" in [uploader.label for uploader in app.file_uploader]
    assert "Key frames" in [number_input.label for number_input in app.number_input]
    assert "EXP-001 start time seconds" in [number_input.label for number_input in app.number_input]
    assert "EXP-002 start time seconds" in [number_input.label for number_input in app.number_input]
    assert "Maximum timing offset seconds" in [
        number_input.label for number_input in app.number_input
    ]

    app.radio[0].set_value("Movement CSVs")
    app.run(timeout=10)

    assert "Reference movement CSV" in [uploader.label for uploader in app.file_uploader]
    assert "Practice movement CSV" in [uploader.label for uploader in app.file_uploader]

    app.radio[0].set_value("Group video")
    app.run(timeout=10)

    assert "Group video" in [uploader.label for uploader in app.file_uploader]
    assert "Reference track ID" in [number_input.label for number_input in app.number_input]
    assert "Group start time seconds" in [number_input.label for number_input in app.number_input]


def test_streamlit_deployment_entrypoint_renders() -> None:
    app = AppTest.from_file("streamlit_app.py")

    app.run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Kinesis Movement Lab"
