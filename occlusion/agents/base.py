"""Common observation bridge. Native model packages are imported lazily."""
import ast
import copy
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

from ..core.geom import yaw_from_quat, wrap180
from ..core.intervention import Intervention
from ..core.rig import Frame, Pose, Rig, View
from ..core import store

TRACK_PREFIX = 'c6002_track_'


def literal(path, name):
    for stmt in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            return ast.literal_eval(stmt.value)
    raise ValueError('missing constant %s in %s' % (name, path))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_rig(model, cfg, depth_bias, depth_max, mask_sizes=None, attach=True):
    """Read cameras from a model config; clone separate co-located tracker depths.

    Old Habitat uses YACS; SimpleVLN uses structured OmegaConf. Only depth
    sensors are added: no existing RGB setting or UUID is mutated. depth_max is the
    range (metres) of the tracker depth cameras, set by the batch; it is independent of
    the depth range the model itself reads.
    """
    depth_max = float(depth_max)
    if not depth_max > 0.:
        raise ValueError('depth_max must be positive')
    modern = model == 'simplevln'
    if modern:
        from omegaconf import OmegaConf, open_dict
        sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors
        items = [(k, s) for k, s in sensors.items() if s.type == 'HabitatSimRGBSensor']
        depths = [s for s in sensors.values() if s.type == 'HabitatSimDepthSensor']
        if not depths:
            raise ValueError('model config has no depth sensor template')
        template = depths[0]
    else:
        sim = cfg.TASK_CONFIG.SIMULATOR
        items = [(k, getattr(sim, k)) for k in sim.AGENT_0.SENSORS
                 if getattr(sim, k).TYPE == 'HabitatSimRGBSensor']
        template = sim.DEPTH_SENSOR
    views, heights, ranges, scales = [], [], [], []
    rgb_keys, depth_keys, tracker_keys = {}, {}, {}
    for key, rgb in items:
        def get(obj, lower, default=None):
            return getattr(obj, lower if modern else lower.upper(), default)
        name = get(rgb, 'uuid') or 'rgb'
        h, w = int(get(rgb, 'height')), int(get(rgb, 'width'))
        orientation = list(get(rgb, 'orientation', [0., 0., 0.]))
        position = list(get(rgb, 'position', [0., 1.25, 0.]))
        if abs(position[0]) > 1e-8 or abs(position[2]) > 1e-8 or abs(orientation[0]) > 1e-8 or abs(orientation[2]) > 1e-8:
            raise ValueError('rig requires centered, upright yaw-only cameras: %s' % name)
        dep = copy.deepcopy(template)
        dh, dw = int(get(dep, 'height')), int(get(dep, 'width'))
        if model == 'omninav':
            dh, dw = h, w
        track_uuid = TRACK_PREFIX + name
        fields = dict(uuid=track_uuid, height=dh, width=dw, hfov=get(rgb, 'hfov'),
                      position=position, orientation=orientation, min_depth=0., max_depth=depth_max,
                      normalize_depth=False)
        if modern:
            with open_dict(dep):
                for field, value in fields.items():
                    setattr(dep, field, value)
            if attach:
                with open_dict(sensors):
                    sensors[track_uuid] = dep
        else:
            for field, value in fields.items():
                setattr(dep, field.upper(), value)
            if attach:
                sensor_key = TRACK_PREFIX.upper() + name.upper()
                setattr(sim, sensor_key, dep)
                if sensor_key not in sim.AGENT_0.SENSORS:
                    sim.AGENT_0.SENSORS.append(sensor_key)
        minimum = float(get(dep, 'min_depth'))
        maximum = float(get(dep, 'max_depth'))
        if minimum != 0.:
            raise ValueError('nonzero MIN_DEPTH cannot be represented by scale-only rig')
        ranges.append(maximum)
        scales.append(maximum if get(dep, 'normalize_depth') else 1.)
        heights.append(float(position[1]))
        target = tuple((mask_sizes or {}).get(name, (h, w)))
        views.append(View(name, float(wrap180(math.degrees(orientation[1]))),
                          float(get(rgb, 'hfov')), (dh, dw), target))
        rgb_keys[name] = name
        depth_keys[name] = 'depth' + (name[3:] if name.startswith('rgb') else '')
        tracker_keys[name] = track_uuid
    if not views or len(set(heights)) != 1 or len(set(ranges)) != 1 or len(set(scales)) != 1:
        raise ValueError('rig needs cameras with a shared height and depth convention')
    return (Rig(model, views, scales[0], ranges[0], heights[0],
                'xyzw' if model == 'etpnav' else 'wxyz', depth_bias),
            rgb_keys, depth_keys, tracker_keys)


class Adapter(object):
    """Concrete adapters own environment initialization and policy stepping."""
    def __init__(self, params, native, maps, output=None):
        self.params, self.native = params, native
        self.rig, self.rgb_keys, self.depth_keys, self.tracker_keys = maps
        self.output = Path(output) if output else None
        self.intervention = None
        self.observation_step = 0
        self.trajectory = []
        self.images = []
        self.cand = None

    def episodes(self):
        return self.episode_map

    def reset(self, ep):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def pose(self):
        state = self.env._sim.get_agent_state()
        q = state.rotation
        return Pose(np.asarray(state.position, dtype=np.float64),
                    yaw_from_quat([q.w, q.x, q.y, q.z], 'wxyz'))

    def metrics(self):
        return dict(self.env.get_metrics())

    def to_frame(self, obs, step):
        from PIL import Image
        rgb, depth, raw = {}, {}, {}
        for view in self.rig.views:
            arr = np.asarray(obs[self.rgb_keys[view.name]])[..., :3]
            if tuple(arr.shape[:2]) != view.mask_hw:
                arr = np.asarray(Image.fromarray(arr).resize((view.mask_hw[1], view.mask_hw[0])))
            rgb[view.name] = arr
            depth[view.name] = self.rig.to_metres(obs[self.tracker_keys[view.name]])
            key = self.depth_keys[view.name]
            if key in obs:
                raw[view.name] = np.asarray(obs[key])
        return Frame(step, self.pose(), rgb, depth, raw or None)

    def from_frame(self, frame, obs):
        result = dict(obs)
        # Never send the independent tracker cameras to model encoders.
        for key in self.tracker_keys.values():
            result.pop(key, None)
        for name, value in frame.rgb.items():
            result[self.rgb_keys[name]] = value
        if frame.depth_raw:
            for name, value in frame.depth_raw.items():
                result[self.depth_keys[name]] = value
        return result

    def observe(self, obs):
        frame = self.to_frame(obs, self.observation_step)
        self.clean_rgb = frame.rgb          # the real observation, before any blackout
        result = self.intervention.on_obs(frame) if self.intervention else frame
        if self.intervention and self.intervention.active(self.observation_step):
            self.save_images(frame, result)
        return self.from_frame(result, obs)

    def save_images(self, clean, result):
        from PIL import Image
        limit = self.params.get('save_images', 0)
        if not self.output or not limit or (limit > 0 and len(self.images) >= limit):
            return
        row = {'step': clean.step, 'views': {}}
        masks = self.intervention.last_masks
        for name in clean.rgb:
            row['views'][name] = {}
            for kind, array in [('clean', clean.rgb[name]), ('black', result.rgb[name]),
                                ('mask', masks[name].astype(np.uint8) * 255)]:
                rel = 'images/%04d-%s-%s.png' % (clean.step, name, kind)
                path = self.output / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(str(path))
                store.record(self.rig.name, 'write', path, None)
                row['views'][name][kind] = rel
        self.images.append(row)

    def begin(self, cand):
        self.cand = cand
        self.observation_step = 0
        self.trajectory, self.images = [], []
        self.intervention = None
        if self.params['arm'] != 'none':
            positions = (store.trajectory_from(cand['model'], cand['split'], cand['ep'], cand['step'])
                         if self.params['region_spec']['kind'] == 'corridor' else None)
            self.intervention = Intervention(cand, self.params['region_spec'], self.params['tracker_spec'],
                                             self.params['follow'], self.params['channels'], self.rig, positions)

    def run(self, cand):
        raise NotImplementedError

    def close(self):
        # Called once per batch, when the run ends or the batch is aborted. The policy is loaded in __init__ and
        # reused across candidates, so it is dropped only here.
        if hasattr(self, 'env'):
            self.env.close()
        if getattr(self, 'agent', None) is not None:
            self.agent = None


def assert_pose(actual, expected, tolerance=1e-4, angle_tolerance=None):
    pos, yaw = expected
    error = float(np.max(np.abs(np.asarray(actual.position) - np.asarray(pos))))
    angle = abs(float(wrap180(actual.yaw - yaw)))
    if error > tolerance or angle > (tolerance if angle_tolerance is None else angle_tolerance):
        raise RuntimeError('replay pose mismatch: position error=%g m yaw error=%g deg' % (error, angle))
    return error
