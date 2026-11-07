"""Region trackers and observation masking (NumPy only, Python 3.6 compatible)."""

import math

import numpy as np

from .geom import clip_arcs, sweep, wrap180
from .regions import Corridor, Heading
from .rig import Frame


def _rows(height, rows):
    height = int(height)
    if rows == 'axis':
        return [height // 2 - 1, height // 2] if height % 2 == 0 else [height // 2]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("rows must be 'axis' or a positive integer")
    return sorted(set(list(range(0, height, rows)) + [height - 1]))


def _merge_intervals(intervals, lo, hi):
    parts = []
    for a, b in intervals:
        a, b = max(float(a), lo), min(float(b), hi)
        if a <= b:
            parts.append([a, b])
    parts.sort()
    out = []
    for a, b in parts:
        if out and a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _point_in_polygon(mask, polygon):
    """Fill a polygon at integer pixel coordinates without a non-NumPy dependency."""
    h, w = mask.shape
    p = np.asarray(polygon, dtype=np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return
    x0 = max(0, int(math.floor(float(p[:, 0].min()))))
    x1 = min(w - 1, int(math.ceil(float(p[:, 0].max()))))
    y0 = max(0, int(math.floor(float(p[:, 1].min()))))
    y1 = min(h - 1, int(math.ceil(float(p[:, 1].max()))))
    if x0 > x1 or y0 > y1:
        return
    xs = np.arange(x0, x1 + 1, dtype=np.float64)[None, :]
    ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None]
    inside = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    boundary = np.zeros_like(inside)
    eps = 1e-8
    for i in range(4):
        x_a, y_a = p[i]
        x_b, y_b = p[(i + 1) % 4]
        dx, dy = x_b - x_a, y_b - y_a
        cross = (xs - x_a) * dy - (ys - y_a) * dx
        on_segment = ((np.abs(cross) <= eps) &
                      (xs >= min(x_a, x_b) - eps) & (xs <= max(x_a, x_b) + eps) &
                      (ys >= min(y_a, y_b) - eps) & (ys <= max(y_a, y_b) + eps))
        boundary |= on_segment
        if abs(dy) > eps:
            crosses = ((y_a > ys) != (y_b > ys)) & (xs < x_a + (ys - y_a) * dx / dy)
            inside ^= crosses
    mask[y0:y1 + 1, x0:x1 + 1] |= inside | boundary


class Tracker(object):
    kind = None

    def __init__(self, rig):
        self.rig = rig
        self.region = None
        self._last_state = {}

    def init(self, region, frame):
        raise NotImplementedError

    def mask(self, frame):
        raise NotImplementedError

    def state(self):
        return dict(self._last_state)


class HeadingTracker(Tracker):
    kind = 'heading'

    def __init__(self, rig, occlusion, rows='axis', incremental=False, unseen_step=None):
        Tracker.__init__(self, rig)
        self.occlusion = bool(occlusion)
        self.rows = rows
        self.incremental = bool(incremental)
        self.unseen_step = None if unseen_step is None else float(unseen_step)
        if self.unseen_step is not None and self.unseen_step <= 0:
            raise ValueError('unseen_step must be positive')
        self.a_heading = np.empty((0,), dtype=np.float64)
        self.a_xz = np.empty((0, 2), dtype=np.float64)
        self.a_range = np.empty((0,), dtype=np.float64)
        self.a_row = np.empty((0,), dtype=np.int64)
        self.a_w = np.empty((0, 3), dtype=np.float64)
        self.covered = []
        self._direction_mask = np.empty((0,), dtype=bool)
        self._n_hidden = 0
        self._last_spans = {}
        self._initialized = False

    def init(self, region, frame):
        if not isinstance(region, Heading):
            raise TypeError('HeadingTracker requires a Heading region')
        if self._initialized:
            raise RuntimeError('tracker is already initialized')
        self.region = region
        self._add(frame)
        self._add_unseen_directions(frame)
        self._initialized = True

    def _append(self, headings, xz, ranges, row_ids, world, direction):
        if not len(headings):
            return
        self.a_heading = np.concatenate((self.a_heading, np.asarray(headings, dtype=np.float64)))
        self.a_xz = np.concatenate((self.a_xz, np.asarray(xz, dtype=np.float64).reshape(-1, 2)))
        self.a_range = np.concatenate((self.a_range, np.asarray(ranges, dtype=np.float64)))
        self.a_row = np.concatenate((self.a_row, np.asarray(row_ids, dtype=np.int64)))
        self.a_w = np.concatenate((self.a_w, np.asarray(world, dtype=np.float64).reshape(-1, 3)))
        self._direction_mask = np.concatenate((self._direction_mask, np.asarray(direction, dtype=bool)))

    def _add(self, frame):
        pose = frame.pose
        low, high = self.region.lo, self.region.hi
        total = 0
        new_coverage = []
        proposed = []
        proposed_keys = set()
        def unseen(h):
            return not any(a - 1e-9 <= h <= b + 1e-9 for a, b in self.covered)
        def propose(view, depth, col, h):
            key = round(h, 7)
            if unseen(h) and key not in proposed_keys:
                proposed.append((view, depth, int(col), float(h)))
                proposed_keys.add(key)
        for view in self.rig.views:
            depth = np.asarray(frame.depth[view.name], dtype=np.float64)
            if depth.shape != tuple(view.depth_hw):
                raise ValueError('depth shape for %s is %r, expected %r' %
                                 (view.name, depth.shape, view.depth_hw))
            h, w = depth.shape
            rel = self.rig.column_heading(view.name, w)
            world_h = self.region.unwrap(float(pose.yaw) + rel)
            owner = self.rig.nearest_view(rel)
            selected = ((world_h >= low - 1e-9) & (world_h <= high + 1e-9) &
                        (owner == self.rig.views.index(view)))
            cols = np.flatnonzero(selected)
            if len(cols):
                angular_step = float(np.median(np.abs(wrap180(np.diff(world_h))))) if w > 1 else view.hfov
                new_coverage.append((float(min(world_h[cols[0]], world_h[cols[-1]]) - angular_step / 2.0),
                                     float(max(world_h[cols[0]], world_h[cols[-1]]) + angular_step / 2.0)))
                for col in cols:
                    hh = float(world_h[col])
                    propose(view, depth, col, hh)
            for endpoint in (low, high):
                rel_endpoint = view.yaw_offset + float(wrap180(
                    endpoint - float(pose.yaw) - view.yaw_offset))
                if (view.yaw_offset - view.hfov / 2.0 - 1e-9 <= rel_endpoint <=
                        view.yaw_offset + view.hfov / 2.0 + 1e-9):
                    if int(self.rig.nearest_view(np.asarray([rel_endpoint]))[0]) != self.rig.views.index(view):
                        continue
                    col = int(np.argmin(np.abs(rel - rel_endpoint)))
                    propose(view, depth, col, float(endpoint))
        if new_coverage:
            self.covered = _merge_intervals(self.covered + [[a, b] for a, b in new_coverage], low, high)
        for view, depth, col, hh in proposed:
            h, w = depth.shape
            rows = _rows(h, self.rows)
            directions, origin = self.rig.rays(view.name, depth.shape, pose)
            for row_index, pixel_row in enumerate(rows):
                z = float(depth[pixel_row, col])
                direction = directions[pixel_row, col]
                hit = np.isfinite(z) and z > 0.0 and z <= float(np.nextafter(np.float32(self.rig.depth_max), np.float32(-np.inf)))
                point = origin + direction * z if hit else direction
                if hit:
                    radius = float(np.linalg.norm((point - origin)[[0, 2]]))
                    a = np.radians(hh)
                    xz = np.asarray(pose.position)[[0, 2]] - radius * np.array([np.sin(a), np.cos(a)])
                else:
                    radius = np.inf
                    xz = np.asarray(pose.position, dtype=np.float64)[[0, 2]]
                self._append([hh], [xz], [radius], [row_index], [point], [not hit])
                total += 1
        if self.incremental:
            self._replace_direction_anchors()
        return total

    def _replace_direction_anchors(self):
        if not np.any(self._direction_mask) or not len(self.a_heading):
            return
        keep = np.ones(len(self.a_heading), dtype=bool)
        real = self.a_row >= 0
        if np.any(real):
            for i in np.flatnonzero(self.a_row == -1):
                if np.min(np.abs(self.a_heading[real] - self.a_heading[i])) < 1e-7:
                    keep[i] = False
        self._select(keep)

    def _select(self, keep):
        self.a_heading = self.a_heading[keep]
        self.a_xz = self.a_xz[keep]
        self.a_range = self.a_range[keep]
        self.a_row = self.a_row[keep]
        self.a_w = self.a_w[keep]
        self._direction_mask = self._direction_mask[keep]

    def _add_unseen_directions(self, frame):
        if self.unseen_step is None:
            return
        low, high = self.region.lo, self.region.hi
        grid = np.arange(low, high + self.unseen_step * 1e-6, self.unseen_step)
        if not len(grid) or grid[-1] < high - 1e-7:
            grid = np.append(grid, high)
        uncovered = np.ones(len(grid), dtype=bool)
        for a, b in self.covered:
            uncovered &= ~((grid >= a - 1e-9) & (grid <= b + 1e-9))
        grid = grid[uncovered]
        if not len(grid):
            return
        ang = np.radians(grid)
        world = np.stack((-np.sin(ang), np.zeros(len(grid)), -np.cos(ang)), axis=1)
        pos = np.asarray(frame.pose.position, dtype=np.float64)
        xz = np.repeat(pos[[0, 2]][None, :], len(grid), axis=0)
        self._append(grid, xz, np.full(len(grid), np.inf), np.full(len(grid), -1),
                     world, np.ones(len(grid), dtype=bool))

    def _visible(self, frame):
        """Whether each anchor is seen from this pose, judged once over all views.

        An anchor is visible if it passes the depth test in any view whose image contains it. One that falls in no
        image is hidden if its bearing is inside the rig's horizontal coverage (it left the vertical field of view),
        and visible if it is outside the coverage (behind the agent): that cannot be judged, so it keeps the polyline
        connected, and the arc is clipped to each view afterwards.
        """
        n = len(self.a_heading)
        if not n:
            return np.empty((0,), dtype=bool)
        is_dir = self._direction_mask
        visible = np.zeros(n, dtype=bool)
        seen = np.zeros(n, dtype=bool)
        for v in self.rig.views:
            depth = np.asarray(frame.depth[v.name], dtype=np.float64)
            seen |= self.rig.project(v.name, self.a_w, is_dir, frame.pose)[3]
            visible |= self.rig.visible(v.name, self.a_w, is_dir, depth, frame.pose)
        origin = np.asarray(frame.pose.position, dtype=np.float64) + [0.0, self.rig.cam_height, 0.0]
        d = np.where(is_dir[:, None], self.a_w, self.a_w - origin)
        bearing = wrap180(np.degrees(np.arctan2(-d[:, 0], -d[:, 2])) - float(frame.pose.yaw))
        covered = np.zeros(n, dtype=bool)
        for v in self.rig.views:
            covered |= np.abs(wrap180(bearing - v.yaw_offset)) <= v.hfov / 2.0
        return visible | (~seen & ~covered)

    def _keep(self, frame):
        keep = np.ones(len(self.a_heading), dtype=bool)
        finite = np.isfinite(self.a_range)
        if np.any(finite):
            here = np.asarray(frame.pose.position, dtype=np.float64)[[0, 2]]
            keep[finite] &= np.linalg.norm(self.a_xz[finite] - here, axis=1) >= 1e-6
        if self.occlusion:
            visible = self._visible(frame)
            self._n_hidden = int(np.sum(keep & ~visible))
            keep &= visible
        else:
            self._n_hidden = 0
        return keep

    def spans(self, frame, view):
        if not self._initialized:
            raise RuntimeError('tracker has not been initialized')
        if self.incremental and getattr(self, '_added_step', None) != frame.step:
            self._add(frame)
            self._added_step = frame.step
        view_cfg = self.rig.view(view)
        arcs = self._arcs(frame)
        view_arcs = clip_arcs(arcs, view_cfg.yaw_offset - view_cfg.hfov / 2.0,
                              view_cfg.yaw_offset + view_cfg.hfov / 2.0)
        self._last_spans[view] = view_arcs
        return view_arcs

    def _arcs(self, frame):
        """The swept bearing arcs for this pose: computed once per frame, then clipped to each view."""
        key = (int(frame.step), tuple(np.asarray(frame.pose.position, dtype=np.float64).tolist()),
               float(frame.pose.yaw), len(self.a_heading))
        if getattr(self, '_arc_key', None) == key:
            return self._arc_cache
        keep = self._keep(frame)
        yaw = float(frame.pose.yaw)
        bearings = np.where(self._direction_mask, self.a_heading,
                            self.region.center + wrap180(
                                np.degrees(np.arctan2(-(self.a_xz[:, 0] - frame.pose.position[0]),
                                                     -(self.a_xz[:, 1] - frame.pose.position[2]))) -
                                self.region.center))
        bearings = wrap180(bearings - yaw)
        rows = sorted(set(int(x) for x in self.a_row if x >= 0))
        groups = []
        for row in (rows or [0]):
            idx = np.flatnonzero((self.a_row == row) | (self.a_row == -1))
            idx = idx[np.argsort(self.a_heading[idx], kind='mergesort')]
            # Split before filtering: invisible anchors break connectivity.
            cuts = np.flatnonzero(~keep[idx])
            groups.extend(x for x in np.split(idx, cuts) if len(x))
        arcs = []
        for group in groups:
            group = group[keep[group]]
            if not len(group):
                continue
            start, width = sweep(bearings[group])
            arcs.append((start, width))
        self._arc_key, self._arc_cache = key, arcs
        return arcs

    def mask(self, frame):
        masks = {}
        for view in self.rig.views:
            h, w = view.mask_hw
            cols = self.rig.column_heading(view.name, w)
            spans = self.spans(frame, view.name)
            col_mask = np.zeros(w, dtype=bool)
            for lo, hi in spans:
                col_mask |= (cols >= lo - 1e-9) & (cols <= hi + 1e-9)
            masks[view.name] = np.broadcast_to(col_mask[None, :], (h, w)).copy()
        self._last_state = self._state_base()      # _n_hidden was counted over the whole rig by _arcs
        self._last_state['spans'] = {key: [list(x) for x in val]
                                     for key, val in self._last_spans.items()}
        return masks

    def _state_base(self):
        finite = np.isfinite(self.a_range)
        return {'n_anchor': int(len(self.a_heading)), 'n_hit': int(finite.sum()),
                'n_hidden': int(self._n_hidden)}


class SurfaceTracker(HeadingTracker):
    kind = 'surface'

    def __init__(self, rig, rows, gap_max, near_z):
        HeadingTracker.__init__(self, rig, occlusion=True, rows=rows,
                                incremental=False, unseen_step=None)
        self.gap_max = float(gap_max)
        self.near_z = float(near_z)
        if self.gap_max <= 0 or self.near_z <= 0:
            raise ValueError('gap_max and near_z must be positive')
        self._n_cell = 0

    def mask(self, frame):
        if not self._initialized:
            raise RuntimeError('tracker has not been initialized')
        masks = {}
        n_cell = 0
        for view in self.rig.views:
            depth = np.asarray(frame.depth[view.name], dtype=np.float64)
            h, w = depth.shape
            out = np.zeros((h, w), dtype=bool)
            if len(self.a_heading):
                is_dir = self._direction_mask
                u, v, zc, inside = self.rig.project(view.name, self.a_w, is_dir, frame.pose)
                vis = self.rig.visible(view.name, self.a_w, is_dir, depth, frame.pose)
                corner_ok = (zc > self.near_z) & (~inside | vis)
                row_ids = sorted(set(int(x) for x in self.a_row if x >= 0))
                by_row = {}
                for row in row_ids:
                    idx = np.flatnonzero(self.a_row == row)
                    idx = idx[np.argsort(self.a_heading[idx], kind='stable')]
                    by_row[row] = idx
                for upper, lower in zip(row_ids[:-1], row_ids[1:]):
                    a, b = by_row[upper], by_row[lower]
                    if len(a) != len(b) or not np.allclose(self.a_heading[a], self.a_heading[b], atol=1e-7):
                        raise RuntimeError('surface anchor rows are not aligned')
                    for j in range(len(a) - 1):
                        corners = np.asarray([a[j], a[j + 1], b[j + 1], b[j]], dtype=np.int64)
                        if self.a_heading[a[j + 1]] - self.a_heading[a[j]] > self.gap_max:
                            continue
                        if not np.all(corner_ok[corners]):
                            continue
                        poly = np.stack((u[corners] - 0.5, v[corners] - 0.5), axis=1)
                        _point_in_polygon(out, poly)
                        n_cell += 1
            rows, cols = self.rig.pixel_map(view.depth_hw, view.mask_hw)
            masks[view.name] = out[rows][:, cols]
        self._n_cell = n_cell
        self._keep(frame)
        self._last_spans = {}
        for view in self.rig.views:
            occupied = np.flatnonzero(masks[view.name].any(axis=0))
            groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
            headings = self.rig.column_heading(view.name, view.mask_hw[1])
            self._last_spans[view.name] = sorted(
                [[float(np.min(headings[g])), float(np.max(headings[g]))]
                 for g in groups if len(g)])
        self._last_state = self._state_base()
        self._last_state['spans'] = self._last_spans
        self._last_state['n_cell'] = int(n_cell)
        return masks


class VolumeTracker(Tracker):
    kind = 'volume'

    def __init__(self, rig, ds):
        Tracker.__init__(self, rig)
        self.ds = float(ds)
        if not np.isfinite(self.ds) or self.ds <= 0:
            raise ValueError('ds must be positive')
        self.n_sample_max = 0

    def init(self, region, frame):
        if not isinstance(region, Corridor):
            raise TypeError('VolumeTracker requires a Corridor region')
        if self.region is not None:
            raise RuntimeError('tracker is already initialized')
        self.region = region

    def mask(self, frame):
        if self.region is None:
            raise RuntimeError('tracker has not been initialized')
        masks = {}
        sample_max = 0
        origin = np.asarray(frame.pose.position, dtype=np.float64) + [0.0, self.rig.cam_height, 0.0]
        here = np.asarray(frame.pose.position, dtype=np.float64)[[0, 2]]
        reach = self.region.reach(origin)
        for view in self.rig.views:
            h, w = view.mask_hw
            source_rays, _ = self.rig.rays(view.name, view.depth_hw, frame.pose)
            rows, cols = self.rig.pixel_map(view.depth_hw, view.mask_hw)
            directions = source_rays[rows][:, cols]
            depth = np.asarray(frame.depth[view.name], dtype=np.float64)[rows][:, cols]
            norms = np.linalg.norm(directions, axis=-1)
            valid_depth = np.isfinite(depth) & (depth > 0.0) & (depth <= float(np.nextafter(np.float32(self.rig.depth_max), np.float32(-np.inf))))
            hit_t = np.where(valid_depth, depth * norms, np.inf)
            max_t = np.minimum(hit_t, reach)
            unit = directions / norms[..., None]
            out = np.zeros((h, w), dtype=bool)
            count = 0
            for distance in np.arange(self.ds, reach + self.ds * 1e-9, self.ds):
                active = (distance <= max_t) & ~out
                if not np.any(active):
                    break
                count += 1
                points = origin + unit[active] * distance
                out[active] = self.region.inside(points, here)
            sample_max = max(sample_max, count)
            hit_rows, hit_cols = np.nonzero(valid_depth)
            if len(hit_rows):
                hit_points = origin + directions[hit_rows, hit_cols] * depth[hit_rows, hit_cols, None]
                out[hit_rows, hit_cols] |= self.region.inside(hit_points, here)
            masks[view.name] = out
        self.n_sample_max = sample_max
        self._last_state = {'n_sample_max': int(sample_max)}
        return masks


def apply(frame, masks, channels):
    channels = tuple(channels)
    unknown = set(channels) - {'rgb', 'depth'}
    if unknown:
        raise ValueError('unknown channels: %s' % sorted(unknown))
    if not channels:
        raise ValueError('at least one channel is required')
    rgb = dict(frame.rgb)
    depth_raw = None if frame.depth_raw is None else dict(frame.depth_raw)
    summary = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if name not in frame.rgb or mask.shape != frame.rgb[name].shape[:2]:
            raise ValueError('mask shape mismatch for RGB view %s' % name)
        if 'rgb' in channels:
            rgb[name] = np.array(frame.rgb[name], copy=True)
            rgb[name][mask] = 0
        if 'depth' in channels:
            if depth_raw is None or name not in depth_raw:
                raise ValueError('raw depth is unavailable for view %s' % name)
            rows, cols = _pixel_map(mask.shape, np.asarray(depth_raw[name]).shape[:2])
            depth_mask = mask[rows][:, cols]
            depth_raw[name] = np.array(depth_raw[name], copy=True)
            if depth_raw[name].ndim == 3 and depth_raw[name].shape[-1] == 1:
                depth_raw[name][..., 0][depth_mask] = 0
            else:
                depth_raw[name][depth_mask] = 0
        summary[name] = {'pixels': int(mask.sum()), 'cols': int(mask.any(axis=0).sum())}
    return frame.copy(rgb=rgb, depth_raw=depth_raw), summary


def _pixel_map(src_hw, dst_hw):
    sh, sw = int(src_hw[0]), int(src_hw[1])
    dh, dw = int(dst_hw[0]), int(dst_hw[1])
    if min(sh, sw, dh, dw) <= 0:
        raise ValueError('image dimensions must be positive')
    rows = np.floor((np.arange(dh) + 0.5) * sh / float(dh)).astype(np.int64)
    cols = np.floor((np.arange(dw) + 0.5) * sw / float(dw)).astype(np.int64)
    return np.clip(rows, 0, sh - 1), np.clip(cols, 0, sw - 1)


def make_tracker(spec, rig):
    if not isinstance(spec, dict) or 'kind' not in spec:
        raise ValueError('tracker spec requires kind')
    cfg = dict(spec)
    kind = cfg.pop('kind')
    if kind == 'heading':
        allowed = {'occlusion', 'rows', 'incremental', 'unseen_step'}
        extra = set(cfg) - allowed
        if extra or 'occlusion' not in cfg:
            raise ValueError('invalid heading tracker options: %s' % sorted(extra))
        return HeadingTracker(rig, **cfg)
    if kind == 'surface':
        allowed = {'rows', 'gap_max', 'near_z'}
        extra = set(cfg) - allowed
        if extra or allowed - set(cfg):
            raise ValueError('invalid surface tracker options: %s' % sorted(extra | (allowed - set(cfg))))
        return SurfaceTracker(rig, **cfg)
    if kind == 'volume':
        if set(cfg) != {'ds'}:
            raise ValueError('volume tracker requires only ds')
        return VolumeTracker(rig, **cfg)
    raise ValueError('unknown tracker kind: %r' % kind)
