from __future__ import annotations

from kinesis.experiments.exp002.group_reference import (
    BodyCenter,
    PoseDetection,
    PoseTrackAssigner,
)


def test_pose_track_assigner_creates_initial_tracks_left_to_right() -> None:
    assigner = PoseTrackAssigner()

    assignments = assigner.assign(
        [
            _detection(x=0.75, signature_offset=0.2),
            _detection(x=0.25, signature_offset=-0.2),
        ],
        frame_index=0,
    )

    assert [(assignment.track_id, assignment.center.x) for assignment in assignments] == [
        (1, 0.25),
        (2, 0.75),
    ]


def test_pose_track_assigner_uses_motion_and_pose_signature_through_crossing() -> None:
    assigner = PoseTrackAssigner()

    first_frame = assigner.assign(
        [
            _detection(x=0.25, signature_offset=-0.2),
            _detection(x=0.75, signature_offset=0.2),
        ],
        frame_index=0,
    )
    second_frame = assigner.assign(
        [
            _detection(x=0.40, signature_offset=-0.2),
            _detection(x=0.60, signature_offset=0.2),
        ],
        frame_index=1,
    )
    crossed_frame = assigner.assign(
        [
            _detection(x=0.40, signature_offset=0.2),
            _detection(x=0.60, signature_offset=-0.2),
        ],
        frame_index=2,
    )

    assert [assignment.track_id for assignment in first_frame] == [1, 2]
    assert [assignment.track_id for assignment in second_frame] == [1, 2]
    assert [(assignment.track_id, assignment.center.x) for assignment in crossed_frame] == [
        (1, 0.60),
        (2, 0.40),
    ]


def _detection(*, x: float, signature_offset: float) -> PoseDetection:
    return PoseDetection(
        pose_landmarks=(),
        center=BodyCenter(x=x, y=0.5),
        pose_signature={
            11: (-0.2 + signature_offset, -0.3),
            12: (0.2 + signature_offset, -0.3),
            23: (-0.2 + signature_offset, 0.3),
            24: (0.2 + signature_offset, 0.3),
        },
    )
