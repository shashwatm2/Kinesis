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

Download the MediaPipe Pose Landmarker model into `models/`:

```bash
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

Run the upload app:

```bash
make exp001-app
```

Or process a video from the command line:

```bash
kinesis exp001 process \
  --input path/to/dance-video.mp4 \
  --model models/pose_landmarker_full.task \
  --output-dir outputs/exp001
```

The command writes:

- `*_pose_overlay.mp4`: processed video with skeleton overlay.
- `keyframe_*.jpg`: representative annotated frames.
- `*_movement_analysis.csv`: frame timestamps, normalized landmark coordinates, visibility scores, raw metrics, quality flags, and smoothed metrics.
- `plot_*.png`: time-series plots for quick visual inspection.
- `run_manifest.json`: the input, configuration, outputs, video metadata, and run summary.

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
