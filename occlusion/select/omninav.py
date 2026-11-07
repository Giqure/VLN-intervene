"""OmniNav: original-instruction th4 at d+offset, never a live forward."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .common import (WORKSPACE, OPEN, candidate, check_source, failures, observation_pose,
                     order, original_metrics, primary, reviews)
from ..core.geom import wrap180


def predictions(directories):
    out = {}
    for directory in directories:
        path = Path(directory)
        index = json.loads((path / 'index.json').read_text(encoding='utf-8'))
        raw = np.load(str(path / 'outputs.npy'), mmap_mode='r')
        values = raw[:, index['conds'].index('true')]
        wp = values[..., 5:15].reshape((-1, 5, 2))
        th4 = np.degrees(np.arctan2(-wp[:, 4, 0], wp[:, 4, 1]))
        for i, row in enumerate(index['rows']):
            key = (str(row['episode']), int(row['step']))
            angle = round(float(th4[i]), 1)
            if key in out and out[key] != angle:
                raise ValueError('conflicting original th4 for %s' % (key,))
            out[key] = angle
    return out


def select(split, offset, rig, options=None):
    options = options or {}
    path, rv = reviews('omninav', options.get('reviews'))
    info = failures('omninav', options.get('failures'), split)
    stats = Path(options.get('stats_dir', OPEN / 'OmniNav/infer_r2r_rxr/data/result_r2r_paths/data/checkpoint-omninav/log'))
    dirs = options.get('predictions', [WORKSPACE / 'interpret/data/swap_dev' / split,
                                       WORKSPACE / 'interpret/data/swap_all' / split])
    pred = predictions(dirs)
    out, skip = [], Counter()
    for review in rv:
        ep = str(review['episode_id'])
        check_source(review, hashlib.sha256((stats / ('stats_%s.json' % ep)).read_bytes()).hexdigest())
        d = review.get('deviation_record_number')
        if d is None:
            skip['no_deviation'] += 1
            continue
        step = int(d) + int(offset)
        rows = primary('omninav', split, ep)
        n = sum(r.get('event') == 'step' for r in rows)
        if not 1 <= step <= n:
            skip['offset_outside_rollout'] += 1
            continue
        if (ep, step) not in pred:
            skip['no_original_output_at_target'] += 1
            continue
        pos, yaw = observation_pose(rows, step)
        th4 = pred[ep, step]
        visible = any(np.any(np.abs(wrap180(rig.column_heading(v.name, v.mask_hw[1]) - th4)) <= 15.)
                      for v in rig.views)
        if not visible:
            skip['heading_not_visible'] += 1
            continue
        out.append(candidate('omninav', split, review, path, info[ep], offset, step,
                             float(wrap180(yaw + th4)),
                             {'th4': th4, 'observation_position': pos, 'observation_yaw': yaw,
                              'orig': original_metrics(rows)}))
    return order(out), dict(skip)
