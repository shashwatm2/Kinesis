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
        +-- csv_export.py writes analysis-ready frame data
        +-- drawing.py overlays a skeleton
        +-- movement_summary.py builds non-medical summary text
        +-- outputs/ stores processed video, CSV, and key frames
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

The CSV currently stores normalized image-space landmarks, not MediaPipe world landmarks. World coordinates can be added later when we have validation data and a clear reason to use them.

## Key Decisions

### Use MediaPipe Tasks Pose Landmarker

The current MediaPipe Python guide recommends the Pose Landmarker Tasks API. It supports image, video, and live stream modes. This experiment uses video mode because MediaPipe can use tracking across frames and because the API accepts frame timestamps.

### Prefer CPU Delegate For The Prototype

EXP-001 explicitly requests MediaPipe's CPU delegate. This is slower than GPU acceleration, but it is more reproducible across local machines, CI, and sandboxed environments. GPU/Metal support can be revisited after the basic pipeline is stable.

### Use Streamlit First

Streamlit gives us a working upload flow quickly. It is not the final product architecture. For EXP-001, the goal is learning and feedback speed, not frontend permanence.

### Keep A CLI

The CLI matters because repeatable processing is easier to test, debug, and automate than clicking through a UI. The Streamlit app and CLI both call the same core processor.

### Keep Analysis Modules Small

EXP-001 keeps analysis code split by responsibility:

- `landmarks.py`: shared MediaPipe landmark access helpers.
- `metrics.py`: deterministic frame-level calculations.
- `csv_export.py`: CSV schema and row construction.
- `movement_summary.py`: human-readable, non-medical summary language.

This makes debugging easier because video IO, metric math, export formatting, and UI presentation can be tested separately.

### Keep Model Files Out Of Git

MediaPipe `.task` files are generated/downloaded assets. They belong in `models/` locally and should not be committed.

## First Milestone

1. Install dependencies.
2. Download `pose_landmarker_full.task`.
3. Run the Streamlit app.
4. Upload a short dance video.
5. Confirm a skeleton overlay appears in the output video or key frames.
6. Download the movement CSV.
7. Review the Movement Summary below the Run Summary.

## Next Learning Steps

- Add a tiny sample video for repeatable integration tests.
- Measure frame rate and processing time.
- Compare `lite`, `full`, and `heavy` model bundles.
- Add confidence overlays and failure states for frames where no pose is detected.
- Add world-coordinate export once there is a clear analysis need.
- Add repeatable sample-video tests with known expected landmark behavior.

## Non-Goals

- Diagnosing injuries.
- Scoring dance quality.
- Providing clinical feedback.
- Multi-person choreography analytics.

Those require stronger validation, domain expertise, and careful product design.
