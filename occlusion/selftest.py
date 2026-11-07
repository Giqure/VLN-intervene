"""No pytest/Habitat needed: run in the ETPNav Python 3.6 environment."""
import json
import numpy as np
from .core.rig import Frame, Pose, Rig, View
from .core.regions import Heading, Corridor, build_angular, check_angular_spec
from .core.trackers import HeadingTracker, SurfaceTracker, VolumeTracker, apply
from .core.geom import quat_from_yaw, yaw_from_quat, wrap180
from .core.stats import cluster_weights
from .core import semantic
from .core.intervention import Intervention


def run():
    checks = []
    rig = Rig('selftest', [View('rgb', 0., 90., (12, 20), (24, 40))], 1., 10., 1., 'xyzw', .02)
    def frame(step=1, yaw=0., depth=4.):
        return Frame(step, Pose(np.zeros(3), yaw), {'rgb': np.full((24, 40, 3), 255, np.uint8)},
                     {'rgb': np.full((12, 20), depth)}, {'rgb': np.full((12, 20, 1), depth)})
    for order in ('wxyz', 'xyzw'):
        for yaw in (-179., 0., 91., 180.):
            assert abs(wrap180(yaw_from_quat(quat_from_yaw(yaw, order), order) - yaw)) < 1e-9
    checks.append('quaternions')
    f = frame()
    t = HeadingTracker(rig, False, 'axis', False, None)
    t.init(Heading(0., 15.), f)
    np.testing.assert_array_equal(t.mask(f)['rgb'][0], np.abs(rig.column_heading('rgb', 40)) <= 15.)
    np.testing.assert_allclose(t.spans(frame(2, 10.), 'rgb'), [[-25., 5.]])
    checks.extend(['stationary_interval', 'rotation'])
    h = Heading(0., 10., 20.)
    assert (h.lo, h.hi) == (-20., 10.) and h.contains(-20.) and h.contains(10.) and not h.contains(10.5)
    t = HeadingTracker(rig, False, 'axis', False, None)
    t.init(h, f)
    ch = rig.column_heading('rgb', 40)
    np.testing.assert_array_equal(t.mask(f)['rgb'][0], (ch >= -20.) & (ch <= 10.))
    r = build_angular({'kind': 'heading', 'bounds': [None, 10., 20.]}, rig, 30., 7.)
    assert (r.lo, r.hi) == (-13., 17.)
    r = build_angular({'kind': 'heading', 'bounds': [5., 10., 20.]}, rig, 30., 7.)
    assert (r.lo, r.hi) == (15., 45.)
    r = build_angular({'kind': 'heading', 'bounds': [10., -20.]}, rig, 30., 7.)
    assert np.allclose((r.lo, r.hi), (10., 40.))
    spec = {'kind': 'columns', 'view': 'rgb', 'bounds': [30, 10]}
    t = HeadingTracker(rig, False, 'axis', False, None)
    t.init(build_angular(spec, rig, 0., None), f)
    want = np.zeros(40, bool)
    want[10:31] = True
    np.testing.assert_array_equal(t.mask(f)['rgb'][0], want)
    np.testing.assert_allclose(t.spans(frame(2, 10.), 'rgb'), np.asarray(t.spans(f, 'rgb')) - 10.)
    for bad in ({'kind': 'heading', 'bounds': [-20., 10.]}, {'kind': 'heading', 'bounds': [0., -1., 5.]},
                {'kind': 'heading', 'bounds': [1., 2., 3., 4.]}, {'kind': 'heading', 'half': 15., 'bounds': [1., 2.]},
                {'kind': 'columns', 'bounds': [1, 2]}, {'kind': 'columns', 'view': 'rgb', 'bounds': [1.5, 2]}):
        try:
            check_angular_spec(bad)
        except ValueError:
            continue
        raise AssertionError('accepted %r' % (bad,))
    checks.append('asymmetric_and_column_regions')
    opt = {'base_url': 'https://x.test', 'api_key_env': 'SELFTEST_KEY', 'model_name': 'm', 'reasoning_effort': 'high',
           'request': 'views', 'parser': 'x_range', 'timeout': 5, 'views': ['rgb'], 'prompt': 'dir={direction} ins={instruction}'}
    semantic.check_options(opt)
    for bad in (dict(opt, timeout=0), dict(opt, views=[]), dict(opt, prompt='dir={direction}'), dict(opt, prompt='no placeholder'), dict(opt, extra=1), dict(opt, base_url='x.test')):
        try:
            semantic.check_options(bad)
        except ValueError:
            continue
        raise AssertionError('accepted %r' % (bad,))
    assert semantic.parse_x_range('view=rgb x_range=[0.25, 0.5]', opt, rig)['bounds'] == [10, 19]
    assert semantic.parse_x_range('x_range=[0.5, 0.25]', opt, rig)['bounds'] == [10, 19]
    for bad in ('no range', 'x_range=[0,1] x_range=[0,1]', 'x_range=[0, 1.5]', 'view=other x_range=[0, 1]'):
        try:
            semantic.parse_x_range(bad, opt, rig)
        except ValueError:
            continue
        raise AssertionError('accepted %r' % bad)
    calls = []
    def fake(options, payload):
        calls.append(payload)
        return 'view=rgb x_range=[0.25, 0.5]', {'model': 'm', 'effort': 'high', 'usage': None}
    real, semantic.call_responses = semantic.call_responses, fake
    try:
        iv = Intervention({'step': 1, 'center': 0., 'extra': {'instruction': 'go'}}, {'kind': 'model', 'options': opt},
                          {'kind': 'heading', 'occlusion': False, 'rows': 'axis', 'incremental': False,
                           'unseen_step': None}, 0, ('rgb',), rig)
        iv.on_obs(f)
    finally:
        semantic.call_responses = real
    assert len(calls) == 1 and iv.log()[0]['region']['spec']['bounds'] == [10, 19]
    assert calls[0][0]['content'][0]['text'] == 'dir=0.0 ins=go' and calls[0][0]['content'][1]['text'] == 'view=rgb hfov=90.0 center=0.0'
    assert iv.log()[0]['region']['request']['direction'] == 0.0
    payload, info = semantic.build_views_input(opt, {'center': 30., 'extra': {'instruction': 'turn'}, 'ep': 'secret', 'category': 'secret'}, frame(1, 10.), rig)
    assert info['direction'] == 20.0 and 'secret' not in json.dumps(payload) and 'ins=turn' in json.dumps(payload)
    mask = iv.last_masks['rgb'][0]
    assert mask[10:20].all() and mask.sum() == 10
    checks.append('model_region')
    t = HeadingTracker(rig, True, 'axis', False, None)
    t.init(Heading(0., 15.), f)
    assert not t.mask(frame(2, depth=1.))['rgb'].any()
    checks.append('occlusion')
    t = SurfaceTracker(rig, rows=2, gap_max=20., near_z=.01)
    t.init(Heading(0., 15.), f)
    m = t.mask(f)
    np.testing.assert_array_equal(m['rgb'], t.mask(frame(2))['rgb'])
    assert m['rgb'].any()
    checks.append('surface_stationary')
    v = VolumeTracker(rig, .1)
    v.init(Corridor([[0., 0.], [0., -5.]], 0., .6, 0., 2., .2), f)
    assert v.mask(f)['rgb'].any()
    checks.append('volume')
    out, _ = apply(f, m, ('rgb', 'depth'))
    assert (f.rgb['rgb'] == 255).all() and not np.array_equal(out.rgb['rgb'], f.rgb['rgb'])
    checks.append('copy_not_mutation')
    w = cluster_weights(['a', 'a', 'b'], 100, 7)
    np.testing.assert_array_equal(w[:, 0], w[:, 1])
    checks.append('cluster_bootstrap')
    return {'passed': len(checks), 'checks': checks, 'numpy': np.__version__}
