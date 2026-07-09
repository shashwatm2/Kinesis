# Kinesis

Kinesis is a long-term software project for building technology that helps people understand, improve, and preserve human movement throughout life.

The first experiment is `EXP-001: Dance Pose Estimation`.

## EXP-001 Goal

Upload a dance video, detect human pose with MediaPipe, overlay a skeleton, display the processed video or representative key frames, and export basic movement measurements for each processed frame.

## Repository Layout

```text
apps/                         Small user-facing apps for experiments
docs/                         Vision notes, experiment plans, and engineering decisions
models/                       Local model files, ignored by Git
src/kinesis/                  Reusable Python package code
tests/                        Automated tests
outputs/                      Local generated outputs, ignored by Git
work/                         Scratch space, ignored by Git
```

The main rule is: reusable logic goes in `src/kinesis`, while notebooks, prototypes, and UI shells call that logic instead of owning it.

## Quick Start

Use Python 3.11 or newer. MediaPipe currently publishes PyPI classifiers through Python 3.12, so use Python 3.12 if Python 3.13 causes installation issues.

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

For local CLI runs, download the MediaPipe Pose Landmarker model into `models/`:

```bash
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

The Streamlit app prepares this model automatically when it is missing, which keeps the binary model file out of the public GitHub repository.

Run the local Streamlit app:

```bash
make app
```

To run the same entrypoint used in deployment:

```bash
streamlit run streamlit_app.py
```

Or process a video from the command line:

```bash
kinesis exp001 process \
  --input path/to/dance-video.mp4 \
  --model models/pose_landmarker_full.task \
  --output-dir outputs/exp001 \
  --max-keyframes 12
```

In the Streamlit app, use the `Key frames` input to choose how many representative annotated frames to save. The app caps this at 60 to avoid accidentally creating very large output folders; the CLI accepts any non-negative `--max-keyframes` value.

Use the start/end time controls when a video has lead-in or ending footage that is not part of the movement you want to analyze. The processed video, movement CSV, summary, plots, and key frames are generated from the selected time window.

The command writes:

- `*_pose_overlay.mp4`: processed video with skeleton overlay.
- `keyframe_*.jpg`: representative annotated frames.
- `*_movement_analysis.csv`: frame timestamps, normalized landmark coordinates, visibility scores, raw metrics, quality flags, and smoothed metrics.
- `plot_*.png`: time-series plots for quick visual inspection.
- `run_manifest.json`: the input, configuration, outputs, video metadata, and run summary.

## EXP-002 Reference Movement Matching

EXP-002 compares a reference movement against a practice movement. In the local app, the primary
workflow accepts two raw videos, processes each through EXP-001, then compares the generated
movement CSVs. It also has a group-video mode that tracks multiple people in one video and compares
each tracked person against a selected reference track. The CLI supports raw videos, group videos,
or existing EXP-001 CSV files.

Raw video comparison:

```bash
kinesis exp002 compare-videos \
  --reference-video path/to/reference.mp4 \
  --practice-video path/to/practice.mp4 \
  --model models/pose_landmarker_full.task \
  --output-dir outputs/exp002/reference-vs-practice
```

Existing CSV comparison:

```bash
kinesis exp002 compare \
  --reference-csv path/to/reference_movement_analysis.csv \
  --practice-csv path/to/practice_movement_analysis.csv \
  --output-dir outputs/exp002/reference-vs-practice
```

Group video reference comparison:

```bash
kinesis exp002 compare-group-video \
  --input path/to/group-video.mp4 \
  --model models/pose_landmarker_full.task \
  --reference-track-id 1 \
  --max-people 4 \
  --output-dir outputs/exp002/group-reference \
  --start-time-seconds 3.0 \
  --end-time-seconds 28.0
```

The score is a movement-signal similarity score, not a dance-quality judgment or medical claim.

## Live Website Deployment

This repository is prepared for Streamlit Community Cloud:

- Entry point: `streamlit_app.py`
- Python dependencies: `requirements.txt`, which installs the local `kinesis` package.
- External Linux packages: `packages.txt`, for OpenCV runtime libraries.
- Streamlit configuration: `.streamlit/config.toml`, including a larger upload limit for dance videos.
- MediaPipe model asset: downloaded at runtime into `models/` if missing, and ignored by Git.

To deploy:

1. Push the latest code to GitHub.
2. Open Streamlit Community Cloud and create a new app from `shashwatm2/Kinesis`.
3. Set the main file path to `streamlit_app.py`.
4. Use Python 3.12 or newer.
5. Deploy.

Do not commit raw videos, generated outputs, model files, or private notes. Uploaded videos are processed by the hosting provider while the app runs, so use public or non-sensitive videos for public demos.

## Engineering Principles

- Keep experiments small, named, and documented.
- Separate product interface code from reusable processing code.
- Prefer repeatable command-line workflows before polishing UI.
- Keep raw values and quality-filtered values side by side so analysis decisions stay inspectable.
- Add tests around deterministic code first; add heavier integration tests when model assets and sample videos are stable.
- Treat movement analysis carefully: visual overlays are helpful feedback, not clinical claims.

## References

- MediaPipe Pose Landmarker Python guide: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- MediaPipe Pose Landmarker model overview: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
