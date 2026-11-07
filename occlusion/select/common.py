"""Read-only inputs and shared candidate contracts. No model imports."""
import gzip
import hashlib
import json
from pathlib import Path

from ..core import store
from ..core.geom import yaw_from_quat

WORKSPACE = Path(__file__).resolve().parents[3]
OPEN = WORKSPACE.parent / 'opensource'
DATASET = WORKSPACE.parent / 'dataset/R2R_VLNCE_v1-3_preprocessed'
EXPORT = WORKSPACE / 'results/export'
REVIEWS = {
    'omninav': EXPORT / 'omninav-val-unseen-failure-analysis/omninav-val-unseen-manual-reviews_1013deviation.json',
    'simplevln': EXPORT / 'simplevln-val-unseen-failure-analysis/simplevln-val-unseen-waypoints-manual-reviews.json',
    'etpnav': EXPORT / 'etpnav-v13-val-unseen-failure-analysis/etpnav-v13-val-unseen-manual-reviews.json',
}
FAILURES = {m: p.parent / 'failures.jsonl' for m, p in REVIEWS.items()}
RULE = 'c6002-v1'


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def reviews(model, path=None):
    path = Path(path or REVIEWS[model])
    data = json.loads(path.read_text(encoding='utf-8'))
    return path, data['reviews']


def failures(model, path=None, split='val_unseen'):
    """{episode_id: record with scene_id and trajectory_id}.

    A missing file is an error (read_jsonl alone would return [] and fail later with a bare KeyError). The SimpleVLN
    export has no failures.jsonl, so there the scene and trajectory come from the R2R episodes it was evaluated on.
    """
    explicit = path is not None
    path = Path(path or FAILURES[model])
    if path.is_file():
        return {str(r['episode_id']): r for r in store.read_jsonl(path)}
    if explicit:
        raise FileNotFoundError('failures file does not exist: %s' % path)
    dataset = DATASET / split / (split + '.json.gz')
    if not dataset.is_file():
        raise FileNotFoundError('no failures.jsonl at %s and no episode table at %s' % (path, dataset))
    with gzip.open(str(dataset), 'rt', encoding='utf-8') as stream:
        episodes = json.load(stream)['episodes']
    return {str(e['episode_id']): {'episode_id': str(e['episode_id']),
                                   'scene_id': str(e['scene_id']).split('/')[-2],
                                   'trajectory_id': e['trajectory_id']} for e in episodes}


def primary(model, split, ep):
    path = store.episode_dir(split, ep) / model / 'primary/rollout.jsonl'
    return store.read_jsonl(path)


def observation_pose(rows, step):
    """One-based action observation: ready, or previous action's AFTER pose."""
    if step == 1:
        row = next(r for r in rows if r.get('event') == 'ready')
    else:
        row = next(r for r in rows if r.get('event') == 'step' and int(r['step']) == step - 1)
    pos = store._position(row)
    if pos is None:
        raise ValueError('observation %s has no position' % step)
    yaw = row.get('yaw')
    if yaw is None:
        yaw = yaw_from_quat(row['rotation'], 'wxyz')
    return pos, float(yaw)


def original_metrics(rows):
    done = next(r for r in reversed(rows) if r.get('event') == 'done')
    return dict(done['metrics'])


def candidate(model, split, review, file, info, offset, step, center, extra):
    rec = dict(model=model, split=split, ep=str(review['episode_id']), scene=str(info['scene_id']),
               trajectory_id=str(info['trajectory_id']), d=int(review['deviation_record_number']),
               d_unit=store.D_UNIT[model], offset=int(offset), step=int(step), center=center,
               category=review['primary_category'], status=review['status'], review_file=str(file),
               source_sha256=review['source_sha256'], rule_version=RULE, extra=extra)
    store.validate(rec, store.CANDIDATE)
    return rec


def check_source(review, actual):
    if review['source_sha256'] != actual:
        raise ValueError('episode %s: source_sha256 mismatch (review=%s, current=%s)' %
                         (review['episode_id'], review['source_sha256'], actual))


def order(candidates):
    return sorted(candidates, key=lambda r: (r['scene'], r['trajectory_id'], r['ep'], r['step']))
