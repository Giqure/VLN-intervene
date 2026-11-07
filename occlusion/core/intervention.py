"""Per-observation intervention hook."""

import numpy as np

from .regions import Corridor, build_angular, check_angular_spec, wrong_path
from .semantic import region_from_model
from .trackers import apply, make_tracker


class Intervention(object):
    def __init__(self, cand, region_spec, tracker_spec, follow, channels, rig,
                 orig_positions=None):
        self.cand = dict(cand)
        self.rig = rig
        self.region_spec = dict(region_spec)
        self.tracker = make_tracker(tracker_spec, rig)
        self.channels = tuple(channels)
        self.start = int(cand['step'])
        self.follow = None if follow is None else int(follow)
        if self.start < 1 or (self.follow is not None and self.follow < 0):
            raise ValueError('step must be positive and follow nonnegative')
        self.end = None if self.follow is None else self.start + self.follow
        self.region = None
        self.region_info = None
        self.records = []
        self.orig_positions = None if orig_positions is None else np.asarray(orig_positions, dtype=np.float64)
        if self.orig_positions is not None and (self.orig_positions.ndim != 2 or
                                                self.orig_positions.shape[1] != 3):
            raise ValueError('orig_positions must have shape [T, 3]')
        if self.region_spec.get('kind') == 'corridor' and self.orig_positions is None:
            raise ValueError('corridor interventions require orig_positions')
        if self.region_spec.get('kind') != 'corridor' and self.orig_positions is not None:
            self.orig_positions = None

    def active(self, step):
        step = int(step)
        return step >= self.start and (self.end is None or step <= self.end)

    def build_region(self, frame):
        cfg = dict(self.region_spec)
        kind = cfg.pop('kind', None)
        if kind == 'model':
            spec, self.region_info = region_from_model(self.region_spec['options'], self.cand, frame, self.rig)
            check_angular_spec(spec)
            return build_angular(spec, self.rig, frame.pose.yaw, self.cand.get('center'))
        if kind in ('heading', 'columns'):
            check_angular_spec(self.region_spec)
            return build_angular(self.region_spec, self.rig, frame.pose.yaw, self.cand.get('center'))
        if kind == 'corridor':
            length = float(cfg.pop('length'))
            still = float(cfg.pop('still'))
            rho = float(cfg.pop('rho'))
            hmin = float(cfg.pop('hmin'))
            hmax = float(cfg.pop('hmax'))
            r_near = float(cfg.pop('r_near'))
            if cfg:
                raise ValueError('unknown corridor region fields: %s' % sorted(cfg))
            return Corridor(wrong_path(self.orig_positions, length, still),
                            float(frame.pose.position[1]), rho, hmin, hmax, r_near)
        raise ValueError('unknown region kind: %r' % kind)

    def on_obs(self, frame):
        if not self.active(frame.step):
            return frame
        if self.region is None:
            if int(frame.step) != self.start:
                raise RuntimeError('first active observation must be intervention step %d, got %d' %
                                   (self.start, int(frame.step)))
            self.region = self.build_region(frame)
            self.tracker.init(self.region, frame)
        masks = self.tracker.mask(frame)
        self.last_masks = masks
        result, views = apply(frame, masks, self.channels)
        record = {'step': int(frame.step),
                  'position': np.asarray(frame.pose.position, dtype=np.float64).tolist(),
                  'yaw': float(frame.pose.yaw), 'views': views,
                  'tracker': self.tracker.state()}
        if self.region_info is not None and not self.records:
            record['region'] = self.region_info
        self.records.append(record)
        return result

    def log(self):
        return list(self.records)
