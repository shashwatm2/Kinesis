# EXP-002: Reference Movement Matching

## Goal

Compare a practice movement recording against a reference movement recording and identify
where the movement is most similar or most different.

This experiment intentionally avoids medical, diagnostic, or injury-risk language. It also avoids
calling one dancer "correct." The first version describes how closely one movement signal matches
another reference signal.

## Success Criteria

- Accept two raw videos:
  - a reference movement video
  - a practice movement video
- Accept one group video with multiple detected people and a selected reference track.
- Process each video through EXP-001 to generate movement CSVs.
- Compare smoothed movement metrics frame by frame.
- Search for a small timing offset so slightly early or delayed movement can still be matched.
- Generate an overall match score.
- Export a per-frame match report CSV.
- Export a summary JSON with the largest difference moments.

## Current Scope

EXP-002 keeps raw-video processing and movement matching as separate steps:

```text
EXP-001 video -> pose + movement CSV
EXP-002 CSV + CSV -> movement match report
```

The app and `compare-videos` CLI command can run those steps together, but the matcher still starts
from CSVs internally. This keeps the algorithm testable without MediaPipe or large video files.

For one-video group comparison, EXP-002 adds a lightweight track-assignment layer:

```text
group video -> multi-pose detection -> track CSV per person -> reference-vs-track reports
```

Track IDs are assigned using motion prediction and body-normalized pose signatures. This is more
stable than left-to-right ordering when dancers cross. It is still not a complete identity tracker:
long occlusions, identical overlapping poses, or people leaving and re-entering the frame can still
cause track mistakes.

## Comparison Signals

The first matcher compares these EXP-001 metrics:

- shoulder height asymmetry
- hip height asymmetry
- torso lean angle
- left/right knee angle
- left/right elbow angle
- center-of-body x-position

It does not compare raw landmark coordinates yet. Joint angles and body-center signals are a
better first comparison layer because they are more interpretable and less tied to body size.

## Time Offset

The matcher searches across a small early/late window. If the best match uses a positive offset,
the practice movement is matching a later timestamp than the reference. In plain language, that
usually means the practice movement appears delayed relative to the reference.

This is a first approximation, not a full choreography timing engine.

## Run From The App

The Streamlit app includes an `EXP-002 Reference Match` tab. The default input type is `Raw videos`.
Upload a reference video and a practice video, adjust the matching settings if needed, and run the
comparison. The app displays the match summary, largest difference moments, processed videos, and
download buttons for the generated report files.

The tab also includes:

- `Group video`: track multiple people in one video and compare each track against a selected
  reference track. The app shows a labeled tracked video and representative tracked key frames so
  track IDs can be checked visually.
- `Movement CSVs`: rerun comparisons without processing videos again.

Raw video and group-video modes include optional start/end time controls. Use these to exclude
lead-in or ending footage before pose processing and matching. In the first implementation, the
two-video workflow applies one shared trim window to both videos.

## Run From The CLI

Raw video comparison:

```bash
kinesis exp002 compare-videos \
  --reference-video outputs/samples/reference.mp4 \
  --practice-video outputs/samples/practice.mp4 \
  --model models/pose_landmarker_full.task \
  --output-dir outputs/exp002/reference-vs-practice \
  --max-offset-seconds 1.0
```

Existing CSV comparison:

```bash
kinesis exp002 compare \
  --reference-csv outputs/exp001/reference/reference_movement_analysis.csv \
  --practice-csv outputs/exp001/practice/practice_movement_analysis.csv \
  --output-dir outputs/exp002/reference-vs-practice \
  --max-offset-seconds 1.0
```

Group video reference comparison:

```bash
kinesis exp002 compare-group-video \
  --input outputs/samples/group.mp4 \
  --model models/pose_landmarker_full.task \
  --reference-track-id 1 \
  --max-people 4 \
  --output-dir outputs/exp002/group-reference
```

The command writes:

- `movement_match_report.csv`: one row per compared frame.
- `movement_match_summary.json`: overall score, best timing offset, and largest difference moments.
- group mode also writes a labeled tracked video, per-track movement CSVs, and
  `group_reference_summary.json`.

## Interpretation

The score is a descriptive similarity score, not a judgment of dance quality. A lower score means
the measured joint-angle and body-position signals differed more from the reference over the
compared frames.

Useful follow-up questions are:

- Which moments had the largest differences?
- Which metric differed most in those moments?
- Was the practice movement consistently early or delayed?
- Were enough frames and enough metrics usable for the result to be meaningful?

## Non-Goals

- Ranking dancers.
- Scoring artistry or musicality.
- Diagnosing movement quality.
- Multi-person tracking.
- Handling very different camera angles.
- Handling mirrored choreography automatically.

Those can become future experiments after the reference-matching core is stable.
