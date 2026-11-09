"""Vertical bounds (v_top / v_bottom) of the surface tracker: pure synthetic geometry, no simulator."""
import numpy as np
import pytest

from occlusion.core.regions import Heading
from occlusion.core.rig import Frame, Pose, Rig
from occlusion.core.trackers import SurfaceTracker, _band_rows, _rows, make_tracker

H, W, HFOV = 120, 160, 90.0
RIG = Rig.from_config('t', dict(views=[dict(name='front', yaw_offset=0.0, hfov=HFOV, depth_hw=(H, W), mask_hw=(H, W))],
                                depth_scale=1.0, depth_max=10.0, cam_height=0.0, quat_order='wxyz', depth_bias=0.01))
POSE = Pose(np.zeros(3), 0.0)


def frame(depth_m, step=1):
    rgb = np.zeros((H, W, 3), np.uint8)
    return Frame(step, POSE, rgb, {'front': np.asarray(depth_m, np.float64)})


def wall(distance=3.0):
    # a flat wall facing the camera: z depth is constant, so the hit point of a ray is distance / cos
    return np.full((H, W), distance)


def tracker(**band):
    t = SurfaceTracker(RIG, rows=8, gap_max=10.0, near_z=0.1, **band)
    t.init(Heading(0.0, 15.0), frame(wall()))
    return t


def angle_of_row(row):
    f = (W / 2.0) / np.tan(np.radians(HFOV / 2.0))
    return np.degrees(np.arctan((H / 2.0 - (row + 0.5)) / f))


def test_band_rows_end_at_the_bounds():
    rows = _band_rows(H, W, HFOV, 8, 20.0, -10.0)
    assert rows == sorted(rows) and len(rows) >= 2
    assert all(-10.0 - 1e-9 <= angle_of_row(r) <= 20.0 + 1e-9 for r in rows)
    # the outermost rows sit within one pixel row of the bounds
    assert angle_of_row(rows[0]) > 20.0 - 1.5 and angle_of_row(rows[-1]) < -10.0 + 1.5
    assert angle_of_row(rows[0] - 1) > 20.0 and angle_of_row(rows[-1] + 1) < -10.0


def test_one_sided_and_unbounded_bands():
    assert _band_rows(H, W, HFOV, 8, None, None) == _rows(H, 8)
    only_top = _band_rows(H, W, HFOV, 8, 0.0, None)
    assert all(angle_of_row(r) <= 1e-9 for r in only_top) and only_top[-1] == H - 1


def test_band_outside_the_image_or_too_thin_is_rejected():
    with pytest.raises(ValueError):
        _band_rows(H, W, HFOV, 8, 80.0, 70.0)
    with pytest.raises(ValueError):
        _band_rows(H, W, HFOV, 8, 0.2, 0.1)


def test_without_bounds_nothing_changes():
    plain = SurfaceTracker(RIG, rows=8, gap_max=10.0, near_z=0.1)
    assert plain.band is None
    plain.init(Heading(0.0, 15.0), frame(wall()))
    mask = plain.mask(frame(wall()))['front']
    cols = np.flatnonzero(mask.any(0))
    assert len(cols) and mask[:, cols].all()          # whole columns, top to bottom, as before
    assert 'band' not in plain.state()


def test_mask_stays_inside_the_vertical_bounds_and_the_azimuth_fan():
    t = tracker(v_top=20.0, v_bottom=-10.0)
    mask = t.mask(frame(wall()))['front']
    rows = np.flatnonzero(mask.any(1))
    assert len(rows)
    assert angle_of_row(rows[0]) <= 20.0 + 1.5 and angle_of_row(rows[-1]) >= -10.0 - 1.5
    assert not mask[: rows[0]].any() and not mask[rows[-1] + 1:].any()
    cols = np.flatnonzero(mask.any(0))
    full = tracker().mask(frame(wall()))['front']
    assert set(cols) <= set(np.flatnonzero(full.any(0)))        # same left/right faces
    assert mask.sum() < full.sum()
    assert t.state()['band'] == [20.0, -10.0]


def test_a_nearer_object_inside_the_band_blocks_its_cells_only():
    t = tracker(v_top=20.0, v_bottom=-10.0)
    blocked = wall()
    blocked[:, : W // 2] = 1.0                                   # nearer than the anchors on the left half
    mask = t.mask(frame(blocked, step=2))['front']
    reference = t.mask(frame(wall(), step=3))['front']
    # depth_ref takes the maximum over each 2x2 neighbourhood, so the seam at the object's edge is one pixel wide
    assert mask.sum() < reference.sum() and not mask[:, : W // 2 - 2].any()
    assert mask[:, W // 2:].any()


def test_tracker_spec_validation_and_plan_time_rig_check():
    spec = dict(kind='surface', rows=8, gap_max=10.0, near_z=0.1)
    assert make_tracker(spec, RIG).band is None
    assert make_tracker(dict(spec, v_top=20.0, v_bottom=-10.0), RIG).band == (20.0, -10.0)
    assert make_tracker(dict(spec, v_top=20.0, v_bottom=None), RIG).band == (20.0, None)
    for bad in (dict(v_top=-10.0, v_bottom=20.0), dict(v_top=95.0), dict(v_bottom='x'), dict(v_top=80.0, v_bottom=70.0)):
        with pytest.raises(ValueError):
            make_tracker(dict(spec, **bad), RIG)
    with pytest.raises(ValueError):
        make_tracker(dict(kind='surface', rows='axis', gap_max=10.0, near_z=0.1, v_top=5.0), RIG)
    with pytest.raises(ValueError):
        make_tracker(dict(kind='heading', occlusion=True, rows=8, incremental=False, unseen_step=None, v_top=5.0), RIG)
