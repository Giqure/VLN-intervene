"""ETPNav ghost provenance comes only from a fully reproduced none run."""
import json
from collections import Counter
from pathlib import Path

from .common import OPEN, candidate, check_source, digest, failures, order, reviews
from ..core.geom import heading

LOGS = OPEN / 'ETPNav/data/logs/eval_results/traj_v13'


def source_logs(options=None):
    options = options or {}
    path = Path(options.get('logs', LOGS))
    return (json.loads((path / 'traj_ckpt_59_val_unseen_r0_w1.json').read_text()),
            json.loads((path / 'stats_ep_ckpt_59_val_unseen_r0_w1.json').read_text()))


def select(split, offset, rig, options=None):
    options = options or {}
    path, rv = reviews('etpnav', options.get('reviews'))
    info = failures('etpnav', options.get('failures'), split)
    trajs, stats = source_logs(options)
    provenance = options.get('provenance')
    out, skip = [], Counter()
    for review in rv:
        ep = str(review['episode_id'])
        t, stat = trajs[ep], stats[ep]
        check_source(review, digest([stat, t]))
        d = review.get('deviation_record_number')
        if d is None:
            skip['no_deviation'] += 1
            continue
        target = int(d) + int(offset)
        dec = t['decisions']
        if not 1 <= target <= len(dec):
            skip['offset_outside_decisions'] += 1
            continue
        if dec[target - 1]['act'] != 4:
            skip['target_is_not_ghost'] += 1
            continue
        step, center = target, None
        extra = {'orig': stat, 'target_planning': target, 'decisions_orig': len(dec)}
        if provenance is not None:
            run = provenance.get(ep)
            if not run or not run.get('checks', {}).get('full_same'):
                skip['none_not_fully_reproduced'] += 1
                continue
            g = run['provenance'][target - 1]
            if not g or not g['fronts'] or not g['pred_pos']:
                raise ValueError('reproduced none has no consumed ghost provenance for %s' % ep)
            step = int(g['fronts'][0])
            if not 1 <= step <= target:
                raise ValueError('invalid ghost discovery planning step')
            center = float(heading(dec[step - 1]['position'], g['pred_pos'][0]))
            extra.update(ghost=g, none_batch=run['batch'])
        # Original position logs are zero-based; next decision starts at previous end - 1.
        position_record = 1 if step == 1 else int(dec[step - 2]['end'])
        extra.update(position_record=position_record,
                     observation_position=dec[step - 1]['position'])
        out.append(candidate('etpnav', split, review, path, info[ep], offset, step, center, extra))
    return order(out), dict(skip)
