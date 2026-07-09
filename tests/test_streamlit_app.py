from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_kinesis_logo_asset_exists() -> None:
    assert Path("assets/kinesis-logo.svg").is_file()


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
