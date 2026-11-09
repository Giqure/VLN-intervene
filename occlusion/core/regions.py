"""World-space intervention regions."""

import numpy as np

from .geom import polyline_dist, wrap180


class Region(object):
    kind = None

    def to_json(self):
        raise NotImplementedError

    @staticmethod
    def from_json(data):
        kind = data.get('kind')
        if kind == 'heading':
            if 'half' in data:
                return Heading(data['center'], data['half'])
            return Heading(data['center'], data['left'], data['right'])
        if kind == 'corridor':
            return Corridor(data['path'], data['floor_y'], data['rho'], data['hmin'],
                            data['hmax'], data['r_near'])
        raise ValueError('unknown region kind: %r' % kind)


class Heading(Region):
    """Bearing interval [center - right, center + left]; positive bearings are to the left."""
    kind = 'heading'

    def __init__(self, center, left, right=None):
        self.center = float(center)
        self.left = float(left)
        self.right = self.left if right is None else float(right)
        if not (np.isfinite(self.center) and np.isfinite(self.left) and np.isfinite(self.right)) \
                or self.left < 0.0 or self.right < 0.0:
            raise ValueError('center must be finite and left/right must be nonnegative')
        self.lo = self.center - self.right
        self.hi = self.center + self.left
        self.band = None          # (v_top, v_bottom) degrees, positive up; set by build_angular from a columns spec with rows

    def unwrap(self, h):
        return self.center + wrap180(np.asarray(h, dtype=np.float64) - self.center)

    def contains(self, h):
        u = self.unwrap(h)
        return (u >= self.lo - 1e-9) & (u <= self.hi + 1e-9)

    def to_json(self):
        data = {'kind': self.kind, 'center': self.center, 'left': self.left, 'right': self.right}
        if self.band is not None:
            data['band'] = list(self.band)
        return data


class Corridor(Region):
    kind = 'corridor'

    def __init__(self, path, floor_y, rho, hmin, hmax, r_near):
        self.path = np.asarray(path, dtype=np.float64)
        if self.path.ndim != 2 or self.path.shape[1] != 2 or len(self.path) < 1:
            raise ValueError('path must have shape [K, 2], K >= 1')
        self.floor_y = float(floor_y)
        self.rho = float(rho)
        self.hmin = float(hmin)
        self.hmax = float(hmax)
        self.r_near = float(r_near)
        if self.rho <= 0 or self.r_near < 0 or self.hmin > self.hmax:
            raise ValueError('invalid corridor dimensions')

    def inside(self, pts, here):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        here = np.asarray(here, dtype=np.float64).reshape(2)
        xz = pts[:, [0, 2]]
        horizontal = np.linalg.norm(xz - here, axis=1)
        height = pts[:, 1] - self.floor_y
        return ((polyline_dist(xz, self.path) <= self.rho) &
                (height >= self.hmin) & (height <= self.hmax) &
                (horizontal >= self.r_near))

    def reach(self, origin):
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        dxz = self.path - origin[[0, 2]]
        horizontal = np.linalg.norm(dxz, axis=1) + self.rho
        vertical = max(abs(self.floor_y + self.hmin - origin[1]),
                       abs(self.floor_y + self.hmax - origin[1]))
        return float(np.sqrt(np.max(horizontal) ** 2 + vertical ** 2))

    def to_json(self):
        return {'kind': self.kind, 'path': self.path.tolist(), 'floor_y': self.floor_y,
                'rho': self.rho, 'hmin': self.hmin, 'hmax': self.hmax,
                'r_near': self.r_near}


def _angle(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError('%s must be a finite number' % label)
    return float(value)


def check_angular_spec(spec):
    """Validate a heading/columns region_spec (no rig needed).

    heading, half   : symmetric about the candidate's th4 centre (kept for existing batches).
    heading, bounds : [c, l, r] -> [c - r, c + l]; c is None/'th4' for the candidate's th4, else degrees in the
                      th4 frame (relative to the robot's forward at the first active observation).
                      [l, r] -> signed degrees in the same frame, interval [r, l] (not centred on th4).
    columns         : view + [l, r] image columns of that view (0-based, inclusive, either order), converted to
                      a world bearing interval at the first active frame and tracked like a heading region.
    """
    kind = spec.get('kind')
    if kind == 'heading' and set(spec) == {'kind', 'half'}:
        if _angle(spec['half'], 'region.half') < 0:
            raise ValueError('region.half must be nonnegative')
    elif kind == 'heading' and set(spec) == {'kind', 'bounds'}:
        b = spec['bounds']
        if not isinstance(b, (list, tuple)) or len(b) not in (2, 3):
            raise ValueError('heading bounds must be [c, l, r] or [l, r]')
        if len(b) == 3:
            if b[0] is not None and b[0] != 'th4':
                _angle(b[0], 'region.bounds[0] (c)')
            if _angle(b[1], 'region.bounds l') < 0 or _angle(b[2], 'region.bounds r') < 0:
                raise ValueError('with [c, l, r], l and r must be nonnegative')
        else:
            if _angle(b[1], 'region.bounds r') > _angle(b[0], 'region.bounds l'):
                raise ValueError('with [l, r], the interval is [r, l], so r <= l is required')
    elif kind == 'columns' and set(spec) in ({'kind', 'view', 'bounds'}, {'kind', 'view', 'bounds', 'rows'}):
        if not isinstance(spec['view'], str) or not spec['view']:
            raise ValueError('columns region needs the view name')
        for key, what in (('bounds', 'columns'), ('rows', 'rows')):
            if key not in spec:
                continue
            b = spec[key]
            if (not isinstance(b, (list, tuple)) or len(b) != 2 or
                    any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in b)):
                raise ValueError('columns region %s must be [a, b], nonnegative integer %s' % (key, what))
    else:
        raise ValueError('angular region has missing or unexpected parameters: %r' % (spec,))


def columns_range(spec, rig):
    """(view name, first column, last column) of a columns spec, checked against the rig."""
    try:
        view = rig.view(spec['view'])
    except KeyError:
        raise ValueError('columns region: unknown view %r (rig has %s)' %
                         (spec['view'], [v.name for v in rig.views]))
    first, last = sorted(spec['bounds'])
    if last >= view.mask_hw[1]:
        raise ValueError('columns region: column %d outside view %s of width %d' %
                         (last, view.name, view.mask_hw[1]))
    return view.name, first, last


def rows_range(spec, rig):
    """(first row, last row) of a columns spec's optional vertical bounds, checked against the rig; None if absent."""
    if 'rows' not in spec:
        return None
    view = rig.view(spec['view'])
    first, last = sorted(spec['rows'])
    if last >= view.mask_hw[0]:
        raise ValueError('columns region: row %d outside view %s of height %d' %
                         (last, view.name, view.mask_hw[0]))
    return first, last


def build_angular(spec, rig, yaw, th4_center):
    """Heading region for a heading/columns spec at the first active frame (world bearings)."""
    yaw = float(yaw)
    if spec['kind'] == 'columns':
        name, first, last = columns_range(spec, rig)
        lo, hi = rig.column_span(name, first, last)
        region = Heading(wrap180(yaw + 0.5 * (lo + hi)), 0.5 * (hi - lo))
        rows = rows_range(spec, rig)
        if rows is not None:
            region.band = rig.row_span(name, rows[0], rows[1])
        return region
    if 'half' in spec:
        if th4_center is None:
            raise ValueError('candidate has no heading center')
        return Heading(th4_center, spec['half'])
    b = spec['bounds']
    if len(b) == 2:
        l, r = b
        return Heading(wrap180(yaw + 0.5 * (l + r)), 0.5 * (l - r))
    c, l, r = b
    if c is None or c == 'th4':
        if th4_center is None:
            raise ValueError('candidate has no heading center')
        return Heading(th4_center, l, r)
    return Heading(wrap180(yaw + c), l, r)


def wrong_path(positions, length, still):
    """Take a length-metre polyline from positions, ignoring sub-still jitter/turns."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 1:
        raise ValueError('positions must have shape [T, 3], T >= 1')
    length, still = float(length), float(still)
    if length <= 0 or still < 0:
        raise ValueError('length must be positive and still nonnegative')
    points = [positions[0, [0, 2]].copy()]
    remaining = length
    last = points[0]
    for p in positions[1:, [0, 2]]:
        delta = p - last
        dist = float(np.linalg.norm(delta))
        if dist < still:
            continue
        if dist >= remaining:
            points.append(last + delta * (remaining / dist))
            remaining = 0.0
            break
        points.append(p.copy())
        remaining -= dist
        last = p
    return np.asarray(points, dtype=np.float64)
