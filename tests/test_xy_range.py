"""Model output with optional row bounds: parser, rows -> vertical angles, and the trackers that accept them."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from occlusion.core.intervention import Intervention
from occlusion.core.regions import build_angular, check_angular_spec, rows_range
from occlusion.core.rig import Frame, Pose, Rig
from occlusion.core.semantic import PARSERS, check_options
from occlusion.runner import validate_params

H, W, HFOV = 120, 160, 90.0
RIG = Rig.from_config('t', dict(views=[dict(name='front', yaw_offset=0.0, hfov=HFOV, depth_hw=(H, W), mask_hw=(H, W)),
                                       dict(name='left', yaw_offset=90.0, hfov=HFOV, depth_hw=(H, W), mask_hw=(H, W))],
                                depth_scale=1.0, depth_max=10.0, cam_height=0.0, quat_order='wxyz', depth_bias=0.01))
OPTIONS = {'views': ['front', 'left']}
CONFIG = Path(__file__).resolve().parents[1] / 'occlusion/configs/omninav-rgb-model-rows.json'
parse = PARSERS['xy_range']


def test_columns_and_rows():
    spec = parse('view=front x_range=[0.25, 0.5] y_range=[0.3, 0.6]', OPTIONS, RIG)
    assert spec == {'kind': 'columns', 'view': 'front', 'bounds': [40, 79], 'rows': [36, 71]}


def test_columns_only_is_allowed_and_gives_no_rows():
    spec = parse('view=front x_range=[0.25, 0.5]', OPTIONS, RIG)
    assert spec == {'kind': 'columns', 'view': 'front', 'bounds': [40, 79]}
    assert PARSERS['x_range']('view=front x_range=[0.25, 0.5]', OPTIONS, RIG) == spec


def test_reversed_y_is_sorted_and_edges_are_kept():
    spec = parse('view=front x_range=[0, 1] y_range=[1, 0]', OPTIONS, RIG)
    assert spec['rows'] == [0, H - 1]


@pytest.mark.parametrize('text', [
    'view=front x_range=[0.2, 0.4] y_range=[0.1, 0.2] y_range=[0.5, 0.6]',     # two y ranges
    'view=front x_range=[0.2, 0.4] y_range=[0.1, 1.2]',                          # outside [0, 1]
    'view=front y_range=[0.1, 0.2]',                                             # no x range
    'view=front x_range=[0.2, 0.4] x_range=[0.5, 0.6] y_range=[0.1, 0.2]',       # two x ranges
])
def test_bad_output_is_an_error(text):
    with pytest.raises(ValueError):
        parse(text, OPTIONS, RIG)


def test_rows_become_vertical_angles_with_the_rig_model():
    spec = {'kind': 'columns', 'view': 'front', 'bounds': [40, 79], 'rows': [H // 2 - 10, H // 2 + 9]}
    check_angular_spec(spec)
    assert rows_range(spec, RIG) == (H // 2 - 10, H // 2 + 9)
    top, bottom = build_angular(spec, RIG, 0.0, None).band
    assert top > 0 > bottom and top == pytest.approx(-bottom)                   # symmetric about the horizon
    focal = (W / 2.0) / math.tan(math.radians(HFOV / 2.0))
    assert top == pytest.approx(math.degrees(math.atan(10.0 / focal)))
    assert build_angular({'kind': 'columns', 'view': 'front', 'bounds': [40, 79]}, RIG, 0.0, None).band is None


def test_rows_outside_the_image_are_rejected():
    with pytest.raises(ValueError):
        rows_range({'kind': 'columns', 'view': 'front', 'bounds': [0, 5], 'rows': [0, H]}, RIG)
    with pytest.raises(ValueError):
        check_angular_spec({'kind': 'columns', 'view': 'front', 'bounds': [0, 5], 'rows': [0, 1, 2]})


def frame(step):
    return Frame(step, Pose(np.zeros(3), 0.0), {'front': np.zeros((H, W, 3), np.uint8), 'left': np.zeros((H, W, 3), np.uint8)},
                 {'front': np.full((H, W), 3.0), 'left': np.full((H, W), 3.0)})


SURFACE = {'kind': 'surface', 'rows': 8, 'gap_max': 10.0, 'near_z': 0.1}


def blackened(spec, tracker=SURFACE):
    iv = Intervention({'step': 1, 'center': 0.0}, spec, tracker, 0, ['rgb'], RIG)
    iv.on_obs(frame(1))
    return iv.last_masks['front']


def test_surface_tracker_blackens_only_the_rows_the_region_asks_for():
    cols = {'kind': 'columns', 'view': 'front', 'bounds': [40, 119]}
    full = blackened(cols)
    banded = blackened(dict(cols, rows=[H // 2 - 10, H // 2 + 9]))
    rows = np.flatnonzero(banded.any(1))
    assert len(rows) and rows[0] >= H // 2 - 12 and rows[-1] <= H // 2 + 11      # within about a pixel of the request
    assert banded.sum() < full.sum() and not (banded & ~full).any()


def test_heading_tracker_refuses_a_region_with_rows_and_surface_refuses_two_sources():
    spec = {'kind': 'columns', 'view': 'front', 'bounds': [40, 119], 'rows': [40, 80]}
    heading = {'kind': 'heading', 'occlusion': True, 'rows': 'axis', 'incremental': False, 'unseen_step': None}
    with pytest.raises(ValueError):
        blackened(spec, heading)
    with pytest.raises(ValueError):
        blackened(spec, dict(SURFACE, v_top=10.0, v_bottom=-10.0))


def test_prompt_and_parser_must_agree_and_the_config_validates():
    params = json.loads(CONFIG.read_text())
    validate_params(params)
    options = params['region_spec']['options']
    check_options(options)
    wrong = dict(options, parser='x_range')
    with pytest.raises(ValueError):
        check_options(wrong)
    heading = dict(params, tracker_spec={'kind': 'heading', 'occlusion': True, 'rows': 'axis',
                                         'incremental': False, 'unseen_step': None})
    with pytest.raises(ValueError):
        validate_params(heading)
