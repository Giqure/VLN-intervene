"""Camera rig and frame types; all depth passed to trackers is expressed in metres."""

import math
from typing import NamedTuple

import numpy as np

from .geom import rot_y, wrap180


class Pose(NamedTuple):
    position: np.ndarray
    yaw: float


class View(NamedTuple):
    name: str
    yaw_offset: float
    hfov: float
    depth_hw: tuple
    mask_hw: tuple


class Frame(object):
    def __init__(self, step, pose, rgb, depth, depth_raw=None):
        self.step = int(step)
        self.pose = pose
        self.rgb = rgb
        self.depth = depth
        self.depth_raw = depth_raw

    def copy(self, rgb=None, depth=None, depth_raw=None):
        return Frame(self.step, self.pose,
                     self.rgb if rgb is None else rgb,
                     self.depth if depth is None else depth,
                     self.depth_raw if depth_raw is None else depth_raw)


class Rig(object):
    def __init__(self, name, views, depth_scale, depth_max, cam_height,
                 quat_order, depth_bias):
        self.name = str(name)
        self.views = list(views)
        self.depth_scale = float(depth_scale)
        self.depth_max = float(depth_max)
        self.cam_height = float(cam_height)
        self.quat_order = str(quat_order)
        self.depth_bias = float(depth_bias)
        if not self.views:
            raise ValueError('rig must contain at least one view')
        if len(set(v.name for v in self.views)) != len(self.views):
            raise ValueError('view names must be unique')
        if (not np.isfinite(self.depth_scale) or not np.isfinite(self.depth_max) or
                self.depth_scale <= 0 or self.depth_max <= 0):
            raise ValueError('depth_scale and depth_max must be finite and positive')
        if self.quat_order not in ('wxyz', 'xyzw'):
            raise ValueError("quat_order must be 'wxyz' or 'xyzw'")
        if not np.isfinite(self.cam_height) or not np.isfinite(self.depth_bias):
            raise ValueError('cam_height and depth_bias must be finite')
        for view in self.views:
            if (not np.isfinite(view.yaw_offset) or not np.isfinite(view.hfov) or
                    not 0.0 < view.hfov < 180.0):
                raise ValueError('view yaw_offset and hfov must be finite; hfov must be in (0, 180)')
            if len(view.depth_hw) != 2 or len(view.mask_hw) != 2 or min(view.depth_hw + view.mask_hw) <= 0:
                raise ValueError('view resolutions must be positive (height, width) pairs')

    @classmethod
    def from_config(cls, name, cfg):
        """Build a serializable rig from normalized config values or parsed sensor config."""
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        view_cfg = cfg.get('views')
        if view_cfg is None:
            sensors = cfg.get('sensors')
            if not isinstance(sensors, dict):
                raise ValueError('config must provide views or sensors')
            view_cfg = []
            for key, item in sensors.items():
                if not key.startswith('rgb') and key not in ('left', 'front', 'right'):
                    continue
                view_cfg.append(dict(name=item.get('name', key.replace('rgb_', '')),
                                     yaw_offset=item['yaw_offset'], hfov=item['hfov'],
                                     depth_hw=item['depth_hw'], mask_hw=item['mask_hw']))
        views = []
        for v in view_cfg:
            views.append(View(str(v['name']), float(v['yaw_offset']), float(v['hfov']),
                              tuple(int(x) for x in v['depth_hw']),
                              tuple(int(x) for x in v['mask_hw'])))
        required = ('depth_scale', 'depth_max', 'cam_height', 'quat_order', 'depth_bias')
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError('missing rig config fields: %s' % ', '.join(missing))
        return cls(name, views, cfg['depth_scale'], cfg['depth_max'], cfg['cam_height'],
                   cfg['quat_order'], cfg['depth_bias'])

    def to_json(self):
        return {'name': self.name, 'views': [dict(name=v.name, yaw_offset=v.yaw_offset,
                    hfov=v.hfov, depth_hw=list(v.depth_hw), mask_hw=list(v.mask_hw)) for v in self.views],
                'depth_scale': self.depth_scale, 'depth_max': self.depth_max,
                'cam_height': self.cam_height, 'quat_order': self.quat_order,
                'depth_bias': self.depth_bias}

    @classmethod
    def from_json(cls, data):
        return cls.from_config(data.get('name', 'unnamed'), data)

    def view(self, name):
        for v in self.views:
            if v.name == name:
                return v
        raise KeyError('unknown view: %s' % name)

    def to_metres(self, raw):
        d = np.asarray(raw, dtype=np.float64)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        if d.ndim != 2:
            raise ValueError('depth must have shape [H, W] or [H, W, 1]')
        d = d * self.depth_scale
        miss = (~np.isfinite(d)) | (d <= 0.0) | (d > float(np.nextafter(np.float32(self.depth_max), np.float32(-np.inf))))
        return np.where(miss, np.inf, d)

    def column_heading(self, view, width):
        v = self.view(view)
        x = (np.arange(int(width), dtype=np.float64) + 0.5) / float(width)
        rel = np.degrees(np.arctan((1.0 - 2.0 * x) * math.tan(math.radians(v.hfov / 2.0))))
        return rel + v.yaw_offset

    def column_span(self, view, first, last):
        """Relative bearing (lo, hi) in degrees covering whole columns first..last (inclusive) of the view's mask.

        Column 0 is the left edge, so a smaller column gives a larger (more leftward) bearing, as in column_heading.
        """
        v = self.view(view)
        width = float(v.mask_hw[1])
        tan_h = math.tan(math.radians(v.hfov / 2.0))
        edge = lambda x: math.degrees(math.atan((1.0 - 2.0 * x / width) * tan_h)) + v.yaw_offset
        return edge(int(last) + 1), edge(int(first))

    def nearest_view(self, rel):
        rel = np.asarray(rel, dtype=np.float64)
        centers = np.asarray([v.yaw_offset for v in self.views], dtype=np.float64)
        delta = np.abs(wrap180(rel[..., None] - centers))
        return np.argmin(delta, axis=-1).astype(np.int64)

    @staticmethod
    def pixel_map(src_hw, dst_hw):
        sh, sw = (int(src_hw[0]), int(src_hw[1]))
        dh, dw = (int(dst_hw[0]), int(dst_hw[1]))
        if min(sh, sw, dh, dw) <= 0:
            raise ValueError('image dimensions must be positive')
        rows = np.floor((np.arange(dh) + 0.5) * sh / float(dh)).astype(np.int64)
        cols = np.floor((np.arange(dw) + 0.5) * sw / float(dw)).astype(np.int64)
        return np.clip(rows, 0, sh - 1), np.clip(cols, 0, sw - 1)

    def rays(self, view, hw, pose):
        v = self.view(view)
        h, w = int(hw[0]), int(hw[1])
        if h <= 0 or w <= 0:
            raise ValueError('image dimensions must be positive')
        tan_h = math.tan(math.radians(v.hfov / 2.0))
        xs = (2.0 * (np.arange(w, dtype=np.float64) + 0.5) / w - 1.0) * tan_h
        ys = (2.0 * (np.arange(h, dtype=np.float64) + 0.5) / h - 1.0) * tan_h * h / float(w)
        local = np.stack(np.broadcast_arrays(xs[None, :], -ys[:, None], -np.ones((1, 1))), axis=-1)
        world = np.dot(local, rot_y(float(pose.yaw) + v.yaw_offset).T)
        origin = np.asarray(pose.position, dtype=np.float64) + np.asarray([0.0, self.cam_height, 0.0])
        return world, origin

    def unproject(self, view, depth, pose):
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError('depth must be a 2D metre array')
        directions, origin = self.rays(view, depth.shape, pose)
        hit = np.isfinite(depth) & (depth > 0.0) & (depth <= float(np.nextafter(np.float32(self.depth_max), np.float32(-np.inf))))
        points = origin + directions * np.where(hit, depth, 0.0)[..., None]
        return points, hit

    def project(self, view, pts, is_dir, pose):
        v = self.view(view)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        is_dir = np.asarray(is_dir, dtype=bool).reshape(-1)
        if len(pts) != len(is_dir):
            raise ValueError('pts and is_dir lengths differ')
        origin = np.asarray(pose.position, dtype=np.float64) + [0.0, self.cam_height, 0.0]
        rel = np.where(is_dir[:, None], pts, pts - origin)
        local = np.dot(rel, rot_y(float(pose.yaw) + v.yaw_offset))
        zc = -local[:, 2]
        th = math.tan(math.radians(v.hfov / 2.0))
        h, w = v.depth_hw
        with np.errstate(divide='ignore', invalid='ignore'):
            u = w / 2.0 * (1.0 + local[:, 0] / (zc * th))
            ytan = th * h / float(w)
            ypix = h / 2.0 * (1.0 - local[:, 1] / (zc * ytan))
        inside = (zc > 0.0) & (u >= 0.0) & (u < w) & (ypix >= 0.0) & (ypix < h)
        return u, ypix, zc, inside

    def depth_ref(self, depth):
        d = np.asarray(depth, dtype=np.float64)
        if d.ndim != 2:
            raise ValueError('depth must be 2D')
        d = np.where((d > 0.0) & (d <= float(np.nextafter(np.float32(self.depth_max), np.float32(-np.inf)))) & np.isfinite(d), d, np.inf)
        padded = np.pad(d, ((0, 1), (0, 1)), mode='edge')
        return np.maximum.reduce((padded[:-1, :-1], padded[1:, :-1],
                                  padded[:-1, 1:], padded[1:, 1:]))

    def visible(self, view, pts, is_dir, depth, pose):
        u, v, zc, inside = self.project(view, pts, is_dir, pose)
        d = self.depth_ref(depth)
        rows = np.clip(np.floor(np.where(inside, v, 0.5) - 0.5).astype(np.int64), 0, d.shape[0] - 1)
        cols = np.clip(np.floor(np.where(inside, u, 0.5) - 0.5).astype(np.int64), 0, d.shape[1] - 1)
        ref = d[rows, cols]
        is_dir = np.asarray(is_dir, dtype=bool).reshape(-1)
        visible = np.where(is_dir, np.isinf(ref), zc <= ref + self.depth_bias)
        return inside & visible
