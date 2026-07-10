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


def test_pose_track_assigner_keeps_ids_after_brief_overlap() -> None:
    assigner = PoseTrackAssigner(max_tracks=2)

    assigner.assign(
        [
            _detection(x=0.25, signature_offset=-0.2),
            _detection(x=0.75, signature_offset=0.2),
        ],
        frame_index=0,
    )
    overlap_frame = assigner.assign(
        [_detection(x=0.50, signature_offset=-0.2)],
        frame_index=1,
    )
    separated_frame = assigner.assign(
        [
            _detection(x=0.25, signature_offset=-0.2),
            _detection(x=0.75, signature_offset=0.2),
        ],
        frame_index=2,
    )

    assert [assignment.track_id for assignment in overlap_frame] == [1]
    assert [(assignment.track_id, assignment.center.x) for assignment in separated_frame] == [
        (1, 0.25),
        (2, 0.75),
    ]


def test_pose_track_assigner_reconnects_locked_tracks_after_dropout() -> None:
    assigner = PoseTrackAssigner(max_tracks=2, max_missing_frames=3)

    assigner.assign(
        [
            _detection(x=0.25, signature_offset=-0.2),
            _detection(x=0.75, signature_offset=0.2),
        ],
        frame_index=0,
    )

    assert assigner.assign([], frame_index=10) == []

    assignments = assigner.assign(
        [
            _detection(x=0.28, signature_offset=-0.2),
            _detection(x=0.72, signature_offset=0.2),
        ],
        frame_index=11,
    )

    assert [(assignment.track_id, assignment.center.x) for assignment in assignments] == [
        (1, 0.28),
        (2, 0.72),
    ]


def test_pose_track_assigner_does_not_create_extra_id_when_locked() -> None:
    assigner = PoseTrackAssigner(max_tracks=2)

    assigner.assign(
        [
            _detection(x=0.25, signature_offset=-0.2),
            _detection(x=0.75, signature_offset=0.2),
        ],
        frame_index=0,
    )
    assignments = assigner.assign(
        [
            _detection(x=0.25, signature_offset=-0.2),
            _detection(x=0.50, signature_offset=0.0),
            _detection(x=0.75, signature_offset=0.2),
        ],
        frame_index=1,
    )

    assert sorted(assignment.track_id for assignment in assignments) == [1, 2]


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
