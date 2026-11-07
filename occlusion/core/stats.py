"""Cluster-bootstrap summaries for paired navigation runs."""

import numpy as np

N_BOOT = 1000
LEVEL = 0.95


def cluster_weights(clusters, n=N_BOOT, seed=0):
    clusters = np.asarray(clusters).reshape(-1)
    n = int(n)
    if n < 1:
        raise ValueError('n must be positive')
    unique, inverse = np.unique(clusters.astype(str), return_inverse=True)
    if not len(unique):
        return np.zeros((n, 0), dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    draws = rng.randint(0, len(unique), size=(n, len(unique)))
    cluster_counts = np.zeros((n, len(unique)), dtype=np.float64)
    rows = np.arange(n)[:, None]
    np.add.at(cluster_counts, (np.broadcast_to(rows, draws.shape), draws), 1.0)
    return cluster_counts[:, inverse]


def mean_ci(v, W, mask=None, level=LEVEL):
    values = np.asarray(v, dtype=np.float64).reshape(-1)
    weights = np.asarray(W, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(values):
        raise ValueError('W must have shape [n_boot, len(v)]')
    valid = np.isfinite(values)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if len(m) != len(values):
            raise ValueError('mask length differs from v')
        valid &= m
    count = int(valid.sum())
    if not count:
        return [None, None, None, 0]
    base = float(values[valid].mean())
    w = weights[:, valid]
    den = w.sum(axis=1)
    numer = np.sum(w * values[valid][None, :], axis=1)
    boot = np.divide(numer, den, out=np.full_like(numer, np.nan), where=den > 0)
    finite = boot[np.isfinite(boot)]
    alpha = (1.0 - float(level)) / 2.0
    if not 0.0 <= alpha < 0.5:
        raise ValueError('level must be in (0, 1]')
    if len(finite):
        lo, hi = np.percentile(finite, [100.0 * alpha, 100.0 * (1.0 - alpha)])
        return [base, float(lo), float(hi), count]
    return [base, None, None, count]


def paired(rows, metrics, W, groups, level=LEVEL):
    rows = list(rows)
    n = len(rows)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] != n:
        raise ValueError('W must have shape [n_boot, len(rows)]')
    out = {}
    for name, group_mask in groups.items():
        mask = np.asarray(group_mask, dtype=bool).reshape(-1)
        if len(mask) != n:
            raise ValueError('group %s mask length differs from rows' % name)
        result = {'n': int(mask.sum())}
        for metric in metrics:
            orig = np.asarray([_metric(r, 'orig', metric) for r in rows], dtype=np.float64)
            branch = np.asarray([_metric(r, 'branch', metric) for r in rows], dtype=np.float64)
            result[metric] = {'orig': mean_ci(orig, W, mask, level),
                              'branch': mean_ci(branch, W, mask, level),
                              'delta': mean_ci(branch - orig, W, mask, level)}
        out[str(name)] = result
    return out


def _metric(row, side, key):
    value = row.get(side, {}).get(key)
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def flips(rows):
    out = {'0->0': 0, '0->1': 0, '1->0': 0, '1->1': 0}
    for row in rows:
        orig = row.get('orig', {}).get('success')
        branch = row.get('branch', {}).get('success')
        if orig is None or branch is None:
            continue
        key = '%d->%d' % (int(bool(orig)), int(bool(branch)))
        out[key] += 1
    return out
