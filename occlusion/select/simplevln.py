"""SimpleVLN waypoint-to-first-visible-frame selection, using archived actions."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .common import (OPEN, candidate, check_source, digest, failures, observation_pose,
                     order, original_metrics, primary, reviews)
from ..core.geom import wrap180

ACTIONS = {'stop': 0, 'move_forward': 1, 'turn_left': 2, 'turn_right': 3}


def source_logs(path):
    path = Path(path)
    # Reuse the evaluator's pure-stdlib validation, never importing its policy.
    import importlib.util
    source = OPEN / 'SimpleVLN/eval/vln/result_store.py'
    spec = importlib.util.spec_from_file_location('c6002_result_store', str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logs = {}
    with (path / 'steps.jsonl').open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                row = json.loads(line)
                key = (row['scene_id'], str(row['episode_id']), row['episode_instruction'],
                       row['trial_id'], row['trial_total'])
                if not isinstance(row.get('steps'), list):
                    continue
                logs[key] = row
            except (ValueError, KeyError, TypeError):
                continue
    result = {}
    for key, row in module.load_result_rows(path, pass_k=1).items():
        if key not in logs:
            raise ValueError('complete result lacks matching step log: %r' % (key,))
        if key[1] in result:
            raise ValueError('ambiguous episode ID across trajectories: %s' % key[1])
        result[key[1]] = (row, logs[key])
    return result


def choose(steps, poses, waypoint, rig, turn_angle):
    actions = [ACTIONS.get(s['action'], s['action']) for s in steps]
    starts = [i for i, a in enumerate(actions) if i == 0 or (a in (2, 3) and actions[i - 1] == 1)]
    if not 1 <= waypoint <= len(starts):
        return None, 'offset_outside_waypoints'
    s = starts[waypoint - 1]
    models = [i for i in range(s + 1) if steps[i]['source'] == 'model']
    if not models:
        return None, 'missing_model_call'
    c, e = max(models), s
    while e < len(steps) and actions[e] in (2, 3):
        e += 1
    if e == s:
        return None, 'no_turn'
    if e == len(steps) or actions[e] != 1:
        return None, 'no_forward_after_turn'
    cs = max(i for i in range(c, e) if steps[i]['source'] == 'model')
    if steps[cs]['plan'][:e - cs] != actions[cs:e]:
        raise ValueError('archived model plan disagrees with actions')
    theta = float(turn_angle) * sum({2: 1, 3: -1}.get(a, 0) for a in actions[cs:e])
    center = float(wrap180(poses[cs][1] + theta))
    b = next((i for i in range(cs, e + 1)
              if any(np.any(np.abs(wrap180(rig.column_heading(v.name, v.mask_hw[1]) + poses[i][1] - center)) <= 15.)
                     for v in rig.views)), None)
    if b is None:
        return None, 'heading_not_visible'
    return {'step': b + 1, 'center': center, 's': s + 1, 'c': c + 1,
            'cs': cs + 1, 'e': e + 1, 'theta': theta}, None


def select(split, offset, rig, options=None):
    options = options or {}
    path, rv = reviews('simplevln', options.get('reviews'))
    info = failures('simplevln', options.get('failures'), split)
    logs = source_logs(options.get('logs', OPEN / 'SimpleVLN/eval/results/simplevln_qwen3vl_4b_steps/r2r_pass1'))
    if 'turn_angle' not in options:
        raise ValueError('selector turn_angle must be read from the model config')
    out, skip = [], Counter()
    for review in rv:
        ep = str(review['episode_id'])
        check_source(review, digest(list(logs[ep])))
        d = review.get('deviation_record_number')
        if d is None:
            skip['no_deviation'] += 1
            continue
        rows = primary('simplevln', split, ep)
        steps = [r for r in rows if r.get('event') == 'step']
        poses = [observation_pose(rows, k + 1) for k in range(len(steps))]
        chosen, reason = choose(steps, poses, int(d) + int(offset), rig, options['turn_angle'])
        if reason:
            skip[reason] += 1
            continue
        b, center = chosen.pop('step'), chosen.pop('center')
        chosen.update(wp=int(d) + int(offset), orig=original_metrics(rows),
                      observation_position=poses[b - 1][0], observation_yaw=poses[b - 1][1])
        out.append(candidate('simplevln', split, review, path, info[ep], offset, b, center, chosen))
    return order(out), dict(skip)
