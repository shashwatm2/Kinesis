# EXP-001: Dance Pose Estimation

## Purpose

Build the smallest useful prototype that proves Kinesis can accept a dance video, detect a human pose, and return visual feedback.

## Success Criteria

- Upload a dance video.
- Detect human pose using MediaPipe.
- Overlay the detected skeleton.
- Display the processed video or key frames.
- Calculate basic movement metrics for each processed frame.
- Export a CSV with timestamps, landmarks, visibility scores, and metrics.

## Architecture

```text
Streamlit app / CLI
        |
        v
VideoPoseProcessor
        |
        +-- OpenCV reads and writes video frames
        +-- MediaPipe Pose Landmarker detects landmarks
        +-- metrics.py calculates frame-level movement measurements
        +-- quality.py marks frame metrics as usable or excluded
        +-- smoothing.py smooths usable metric values
        +-- csv_export.py writes analysis-ready frame data
        +-- drawing.py overlays a skeleton
        +-- movement_summary.py builds non-medical summary text
        +-- plots.py writes time-series plot images
        +-- manifest.py writes run metadata
        +-- outputs/ stores processed video, CSV, plots, manifest, and key frames
```

The app is intentionally thin. It handles upload and display only. The processing logic lives in `src/kinesis/experiments/exp001`, which makes it testable and reusable from a CLI, notebook, API, or future product UI.

## Current Movement Metrics

EXP-001 records descriptive 2D, normalized image-space measurements from the primary detected pose in each processed frame.

The current metrics are:

- `shoulder_height_asymmetry`: absolute vertical difference between left and right shoulder y-coordinates.
- `hip_height_asymmetry`: absolute vertical difference between left and right hip y-coordinates.
- `torso_lean_angle_degrees`: angle of the line from hip center to shoulder center relative to vertical. Positive means the shoulder center is to image-right of the hip center.
- `left_knee_angle_degrees` and `right_knee_angle_degrees`: 2D angle at the knee using hip, knee, and ankle landmarks.
- `left_elbow_angle_degrees` and `right_elbow_angle_degrees`: 2D angle at the elbow using shoulder, elbow, and wrist landmarks.
- `center_of_body_x`: average x-position of the visible shoulder and hip landmarks.
- `average_landmark_visibility` and `average_landmark_presence`: average MediaPipe landmark scores for the frame.

These metrics are not medical measurements, diagnoses, or injury-risk indicators. They are basic research signals for understanding what the pose pipeline can measure consistently.

## CSV Export

Each processed frame produces one CSV row.

The CSV includes:

- `frame_index`, `timestamp_ms`, `timestamp_seconds`, and `pose_detected`.
- One set of normalized raw landmark columns for each MediaPipe pose landmark: `x`, `y`, `z`, `visibility`, and `presence`.
- The calculated movement metric columns listed above.

EXP-001.1 adds quality and smoothing columns:

- `*_quality`: whether the raw metric passed basic usability checks.
- `*_quality_reason`: why the metric was kept or excluded.
- `quality_usable_metric_count`: how many metrics were usable in that frame.
- `smoothed_*`: moving-average values calculated only from usable metric values.

The CSV currently stores normalized image-space landmarks, not MediaPipe world landmarks. World coordinates can be added later when we have validation data and a clear reason to use them.

## Quality Control

The first version of movement analysis intentionally uses simple, inspectable quality rules:

- Required landmarks must be present and above the landmark visibility threshold.
- Average landmark visibility must pass `metric_min_average_visibility`.
- Values must stay inside broad image-space plausibility bounds.
- Sudden frame-to-frame jumps are excluded for metrics where a single-frame spike is usually less useful than a stable trend.

These rules are not scientific validation. They are engineering guardrails that prevent obvious landmark failures from dominating plots and summaries.

## Smoothing

EXP-001.1 applies a moving average to quality-passing metric values. Raw values remain in the CSV, and smoothed values are written separately. This is intentional: raw values are needed for debugging; smoothed values are better for human-readable summaries and plots.

Default smoothing is `5` processed frames.

## Run Manifest

Every processed run writes `run_manifest.json`.

The manifest records:

- input path and file name
- output video, CSV, key frame, plot, and manifest paths
- reported frame count, processed frame count, FPS, width, and height
- MediaPipe and analysis configuration
- movement summary values and metric usage counts

## Key Decisions

### Use MediaPipe Tasks Pose Landmarker

The current MediaPipe Python guide recommends the Pose Landmarker Tasks API. It supports image, video, and live stream modes. This experiment uses video mode because MediaPipe can use tracking across frames and because the API accepts frame timestamps.

### Prefer CPU Delegate For The Prototype

EXP-001 explicitly requests MediaPipe's CPU delegate. This is slower than GPU acceleration, but it is more reproducible across local machines, CI, and sandboxed environments. GPU/Metal support can be revisited after the basic pipeline is stable.

### Use Streamlit First

Streamlit gives us a working upload flow quickly. It is not the final product architecture. For EXP-001, the goal is learning and feedback speed, not frontend permanence.

The app exposes a bounded `Key frames` input for saving representative annotated frames. This is a UI guardrail: the core processor accepts any non-negative key-frame count, while the app caps the value to avoid accidentally generating too many image files during exploratory runs.

The app also exposes optional start/end time controls. These define the movement window to process
when the beginning or ending of the video is not part of the dance. All generated artifacts use that
selected window: overlay video, movement CSV, plots, summary, and key frames.

### Keep A CLI

The CLI matters because repeatable processing is easier to test, debug, and automate than clicking through a UI. The Streamlit app and CLI both call the same core processor.

### Keep Analysis Modules Small

EXP-001 keeps analysis code split by responsibility:

- `landmarks.py`: shared MediaPipe landmark access helpers.
- `metrics.py`: deterministic frame-level calculations.
- `quality.py`: metric usability rules.
- `smoothing.py`: moving-average smoothing over usable values.
- `csv_export.py`: CSV schema and row construction.
- `movement_summary.py`: human-readable, non-medical summary language.
- `plots.py`: plot generation for quick visual review.
- `manifest.py`: repeatable run metadata.

This makes debugging easier because video IO, metric math, export formatting, and UI presentation can be tested separately.

### Keep Model Files Out Of Git

MediaPipe `.task` files are generated/downloaded assets. They belong in `models/` locally and should not be committed. The Streamlit app can download the public Pose Landmarker model at runtime so a fresh deployment does not depend on a private laptop file.

## First Milestone

1. Install dependencies.
2. Download `pose_landmarker_full.task` for CLI runs, or let the Streamlit app prepare it.
3. Run the Streamlit app.
4. Upload a short dance video.
5. Confirm a skeleton overlay appears in the output video or key frames.
6. Download the movement CSV.
7. Review the Movement Summary below the Run Summary.
8. Inspect the metric plots.
9. Save the run manifest with any notes about the video and setup.

## Next Learning Steps

- Add a tiny sample video for repeatable integration tests.
- Measure frame rate and processing time.
- Compare `lite`, `full`, and `heavy` model bundles.
- Add confidence overlays and failure states for frames where no pose is detected.
- Add world-coordinate export once there is a clear analysis need.
- Add repeatable sample-video tests with known expected landmark behavior.
- Add a review workflow for marking suspicious metric regions in a run.

## Non-Goals

- Diagnosing injuries.
- Scoring dance quality.
- Providing clinical feedback.
- Multi-person choreography analytics.

Those require stronger validation, domain expertise, and careful product design.
