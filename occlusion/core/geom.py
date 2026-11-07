"""NumPy-only geometry helpers for occlusion experiments (Python 3.6 compatible)."""

import math

import numpy as np


def wrap180(a):
    """Wrap degrees to (-180, 180], preserving scalar/array shape."""
    out = (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0
    out = np.where(out <= -180.0, 180.0, out)
    if np.ndim(a) == 0:
        return float(out)
    return out


def yaw_from_quat(q, order):
    """Return world heading from a pure-yaw quaternion in wxyz or xyzw order."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError('quaternion must have shape (4,)')
    if order == 'wxyz':
        w, x, y, z = q
    elif order == 'xyzw':
        x, y, z, w = q
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError('quaternion must be finite and nonzero')
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return float(math.degrees(2.0 * math.atan2(y, w)))


def quat_from_yaw(yaw, order):
    half = math.radians(float(yaw)) * 0.5
    q = [math.cos(half), 0.0, math.sin(half), 0.0]
    if order == 'wxyz':
        return q
    if order == 'xyzw':
        return [q[1], q[2], q[3], q[0]]
    raise ValueError("order must be 'wxyz' or 'xyzw'")


def rot_y(deg):
    """Column-vector rotation around y; forward -z turns left for positive angles."""
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def heading(p, q):
    """World heading from p to q in the xz plane, with 0 toward world -z."""
    d = np.asarray(q, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    if d.shape[-1] < 3:
        raise ValueError('p and q must have at least three coordinates')
    return float(math.degrees(math.atan2(-float(d[..., 0]), -float(d[..., 2]))))


def polyline_dist(q, path):
    """Minimum xz-plane distance from each point in q to a polyline."""
    q = np.asarray(q, dtype=np.float64)
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 1:
        raise ValueError('path must have shape [K, 2], K >= 1')
    if q.shape[-1] != 2:
        raise ValueError('q must end in dimension 2')
    if len(path) == 1:
        return np.linalg.norm(q - path[0], axis=-1)
    best = np.full(q.shape[:-1], np.inf, dtype=np.float64)
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        den = max(float(np.dot(ab, ab)), 1e-15)
        t = np.clip(np.sum((q - a) * ab, axis=-1) / den, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(q - (a + t[..., None] * ab), axis=-1))
    return best


def path_heading(pos, ref, lam):
    """Heading from the closest point on a reference path, advanced by lam metres."""
    pos = np.asarray(pos, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pos.shape != (3,) or ref.ndim != 2 or ref.shape[1] != 3 or len(ref) < 1:
        raise ValueError('pos must be (3,) and ref must be [M, 3], M >= 1')
    if len(ref) == 1:
        return heading(pos, ref[0])
    lam = max(float(lam), 0.0)
    xy = ref[:, [0, 2]]
    q = pos[[0, 2]]
    seg = xy[1:] - xy[:-1]
    lens = np.linalg.norm(seg, axis=1)
    nonzero = np.flatnonzero(lens > 1e-12)
    if not len(nonzero):
        return heading(pos, ref[-1])
    den = np.maximum(lens ** 2, 1e-15)
    t = np.clip(np.sum((q - xy[:-1]) * seg, axis=1) / den, 0.0, 1.0)
    proj = xy[:-1] + t[:, None] * seg
    dist = np.linalg.norm(proj - q, axis=1)
    dist[lens <= 1e-12] = np.inf
    i = int(np.argmin(dist))
    if lam <= 1e-12:
        delta = seg[i]
        return float(math.degrees(math.atan2(-delta[0], -delta[1])))
    remaining = lam
    j = i
    first_available = (1.0 - t[i]) * lens[i]
    if first_available > 1e-12:
        if remaining <= first_available:
            point = proj[i] + seg[i] * (remaining / lens[i])
            return float(math.degrees(math.atan2(-(point[0] - q[0]), -(point[1] - q[1]))))
        remaining -= first_available
    j += 1
    while j < len(seg):
        if lens[j] <= 1e-12:
            j += 1
            continue
        if remaining <= lens[j]:
            point = xy[j] + seg[j] * (remaining / lens[j])
            return float(math.degrees(math.atan2(-(point[0] - q[0]), -(point[1] - q[1]))))
        remaining -= lens[j]
        j += 1
    end = xy[-1]
    if np.linalg.norm(end - q) > 1e-12:
        return float(math.degrees(math.atan2(-(end[0] - q[0]), -(end[1] - q[1]))))
    last = int(nonzero[-1])
    return float(math.degrees(math.atan2(-seg[last, 0], -seg[last, 1])))


def sweep(b):
    """Return the unwrapped arc (start, width) swept in the supplied order."""
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if not len(b):
        raise ValueError('sweep requires at least one bearing')
    u = b[0] + np.concatenate(([0.0], np.cumsum(wrap180(np.diff(b)))))
    lo, hi = float(np.min(u)), float(np.max(u))
    return float(wrap180(lo)), hi - lo


def clip_arcs(arcs, lo, hi):
    """Intersect circular arcs with a non-wrapping view interval and merge overlaps."""
    lo, hi = float(lo), float(hi)
    if hi < lo:
        raise ValueError('hi must be >= lo')
    pieces = []
    for start, width in arcs:
        start, width = float(start), float(width)
        if width < 0:
            raise ValueError('arc width must be nonnegative')
        if width >= 360.0:
            return [[lo, hi]]
        # Keep the view's unwrapped angular branch (rear cameras commonly
        # occupy [135, 225]); clipping only in [-180, 180] loses half a band.
        s = float(wrap180(start))
        first = int(math.ceil((lo - s - width) / 360.0))
        last = int(math.floor((hi - s) / 360.0))
        for turn in range(first, last + 1):
            a, b = max(s + 360.0 * turn, lo), min(s + width + 360.0 * turn, hi)
            if a <= b:
                pieces.append([a, b])
    pieces.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for a, b in pieces:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged
