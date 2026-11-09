"""Plan/run/analyze orchestration; policy imports are deliberately lazy."""
import contextlib
import datetime
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import subprocess
from collections import Counter
import time
import traceback
from pathlib import Path

import numpy as np

from .core import semantic, stats, store
from .core.regions import check_angular_spec, columns_range
from .core.rig import Rig
from .core.trackers import make_tracker
from .paths import REPO_ROOT, RESULTS, WORKSPACE
from .select.common import original_metrics, primary

ADAPTERS = {'omninav': 'OmniNavAdapter', 'simplevln': 'SimpleVLNAdapter', 'etpnav': 'ETPNavAdapter'}
METRICS = ('success', 'spl', 'ndtw')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_params(params):
    """No implicit geometry defaults, including explicit None for unseen_step."""
    for field in ('model', 'split', 'arm', 'region_spec', 'tracker_spec', 'follow',
                  'offset', 'channels', 'depth_bias', 'depth_max', 'adapter'):
        if field not in params:
            raise ValueError('batch config requires %s' % field)
    if params['model'] not in ADAPTERS or params['arm'] not in ('none', 'rgb', 'rgbd'):
        raise ValueError('unknown model/arm')
    if params['arm'] == 'rgbd' and params['model'] != 'etpnav':
        raise ValueError('only ETPNav supports rgbd')
    channels = list(params['channels'])
    expected = {'none': [], 'rgb': ['rgb'], 'rgbd': ['rgb', 'depth']}[params['arm']]
    if sorted(channels) != sorted(expected):
        raise ValueError('channels must match arm: %s' % expected)
    for name in ('offset', 'save_images'):
        value = params.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('%s must be integer' % name)
    follow = params['follow']
    if follow is not None and (isinstance(follow, bool) or not isinstance(follow, int) or follow < 0):
        raise ValueError('follow must be null or a nonnegative integer')
    def numeric(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('%s must be a finite number (replace template placeholders)' % label)
        return value
    if numeric(params['depth_bias'], 'depth_bias') < 0:
        raise ValueError('depth_bias must be nonnegative')
    if numeric(params['depth_max'], 'depth_max') <= 0:
        raise ValueError('depth_max must be positive (metres, range of the tracker depth cameras)')
    region, tracker = params['region_spec'], params['tracker_spec']
    required = {'corridor': ('rho', 'hmin', 'hmax', 'r_near', 'length', 'still')}
    kind = region.get('kind')
    if kind in ('heading', 'columns'):
        check_angular_spec(region)
    elif kind == 'model':
        if set(region) != {'kind', 'options'}:
            raise ValueError('model region needs exactly kind and options')
        semantic.check_options(region['options'])
    else:
        if kind not in required or set(region) != set(required[kind]) | {'kind'}:
            raise ValueError('region has missing or unexpected parameters')
        for key in required[kind]:
            numeric(region[key], 'region.' + key)
    if kind == 'corridor' and not (region['rho'] > 0 and region['hmax'] > region['hmin'] and
                                   region['r_near'] >= 0 and region['length'] > 0 and region['still'] >= 0):
        raise ValueError('invalid corridor geometry')
    allowed = {'heading': ('occlusion', 'rows', 'incremental', 'unseen_step'),
               'surface': ('rows', 'gap_max', 'near_z'), 'volume': ('ds',)}
    tk = tracker.get('kind')
    if tk not in allowed or set(tracker) != set(allowed[tk]) | {'kind'}:
        raise ValueError('tracker has missing or unexpected parameters')
    if (kind == 'corridor') != (tk == 'volume'):
        raise ValueError('heading pairs with heading/surface; corridor pairs with volume')
    for key in ('occlusion', 'incremental'):
        if key in tracker and not isinstance(tracker[key], bool):
            raise ValueError('tracker.%s must be bool' % key)
    for key in ('gap_max', 'near_z', 'ds', 'unseen_step'):
        if key in tracker and not (key == 'unseen_step' and tracker[key] is None):
            if numeric(tracker[key], 'tracker.' + key) <= 0:
                raise ValueError('tracker.%s must be positive' % key)
    if tk == 'surface' and tracker['rows'] == 'axis':
        raise ValueError('surface rows must be a positive integer')
    if 'rows' in tracker and tracker['rows'] != 'axis':
        if isinstance(tracker['rows'], bool) or not isinstance(tracker['rows'], int) or tracker['rows'] <= 0:
            raise ValueError('rows must be axis or positive integer')
    if 'scratch' not in params['adapter'] or not Path(params['adapter']['scratch']).is_absolute():
        raise ValueError('adapter.scratch must be an absolute isolated output directory')
    if params['model'] == 'simplevln':
        for key in ('base_url', 'model_name', 'max_history', 'gt_path'):
            if key not in params['adapter']:
                raise ValueError('SimpleVLN adapter requires %s' % key)
        history = params['adapter']['max_history']
        if isinstance(history, bool) or not isinstance(history, int) or history < 1:
            raise ValueError('SimpleVLN adapter.max_history must be a positive integer')
    if params['model'] == 'omninav':
        if 'model_path' not in params['adapter']:
            raise ValueError('OmniNav adapter requires model_path')
        if params['adapter'].get('history') not in ('clean', 'blackened'):
            raise ValueError("OmniNav adapter.history must be 'clean' (history keeps the real observation) "
                             "or 'blackened' (history keeps what the model was shown)")
    return params


def configure(params):
    params = json.loads(json.dumps(validate_params(params)))
    module = importlib.import_module('.agents.' + params['model'], __package__)
    native, maps = module.configuration(params, attach=False)
    params['rig'] = maps[0].to_json()
    make_tracker(params['tracker_spec'], maps[0])
    if params['region_spec']['kind'] == 'columns':
        columns_range(params['region_spec'], maps[0])
    if params['region_spec']['kind'] == 'model':
        semantic.check_rig(params['region_spec']['options'], maps[0])
    if params['model'] == 'simplevln':
        params.setdefault('selection', {})['turn_angle'] = float(native[1].habitat.simulator.turn_angle)
    return params


def batch_dir(model, batch):
    return store.experiment_dir(model) / store._component(batch, 'batch')


@contextlib.contextmanager
def exclusive(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('another process holds lock: %s' % path)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_batch(model, batch):
    root = batch_dir(model, batch)
    params = json.loads((root / 'params.json').read_text(encoding='utf-8'))
    validate_params(params)
    if store.batch_name(params) != batch or params['model'] != model:
        raise ValueError('batch identity differs from archived params')
    candidates = store.read_jsonl(root / 'candidates.jsonl')
    for c in candidates:
        store.validate(c, store.CANDIDATE)
        if c['model'] != model or c['split'] != params['split'] or c['offset'] != params['offset']:
            raise ValueError('candidate does not match batch')
    return params, candidates


def plan(params, none_batch=None):
    store._archive_available(metadata=True)  # never create/fall back to another archive
    params = configure(params)
    model, split = params['model'], params['split']
    options = dict(params.get('selection', {}))
    if none_batch:
        if model != 'etpnav' or params['arm'] == 'none':
            raise ValueError('--none-batch is for ETPNav intervention planning only')
        np_, nc = load_batch(model, none_batch)
        if np_['arm'] != 'none' or np_['split'] != split or np_['offset'] != params['offset']:
            raise ValueError('none batch split/offset/arm mismatch')
        provenance = {}
        for c in nc:
            p = store.run_path(model, split, c['ep'], c['step'], none_batch)
            if p.is_file():
                r = json.loads(p.read_text())
                store.validate(r, store.RUN)
                provenance[c['ep']] = r
        params.setdefault('selection', {})['none_batch'] = none_batch
        options['provenance'] = provenance
    if model == 'etpnav' and params['arm'] != 'none' and not none_batch:
        raise ValueError('ETPNav intervention plan requires --none-batch')
    module = importlib.import_module('.select.' + model, __package__)
    candidates, skipped = module.select(split, params['offset'], Rig.from_json(params['rig']), options)
    batch = store.batch_name(params)
    root = batch_dir(model, batch)
    with exclusive(store.experiment_dir(model) / '.plan.lock'):
        if (root / 'params.json').exists():
            old_params, old_candidates = load_batch(model, batch)
            if old_candidates != candidates or old_params != params:
                raise ValueError('existing immutable batch differs; do not overwrite')
            return dict(batch=batch, candidates=len(candidates), skipped=skipped, resumed=True)
        for c in candidates:
            rows = primary(model, split, c['ep'])
            existing = next((r for r in rows if r.get('event') == 'deviation'), None)
            fields = {k: c[k] for k in ('d', 'd_unit', 'category', 'status', 'review_file', 'source_sha256')}
            conversion = {k: c[k] for k in ('offset', 'step', 'center', 'rule_version')}
            conversion.update(batch=batch, position_record=c['extra'].get('position_record'))
            if existing:
                for k in ('d', 'd_unit', 'source_sha256'):
                    if existing.get(k) != fields[k]:
                        raise ValueError('existing deviation source differs for %s' % c['ep'])
                conversions = list(existing.get('conversions', []))
                if conversion not in conversions:
                    conversions.append(conversion)
                fields.update(conversion, conversions=conversions)
                if any(existing.get(k) != v for k, v in fields.items()):
                    store.update_deviation(split, c['ep'], model, fields)
            else:
                fields.update(conversion, conversions=[conversion])
                store.append_deviation(split, c['ep'], model, fields)
        store.write_json_atomic(root / 'params.json', params)
        path = root / 'candidates.jsonl'
        before = store._hash(path) if path.exists() else None
        store._atomic_bytes(path, b''.join(store._json_bytes(c) for c in candidates))
        store.record(model, 'write', path, before)
        store.write_json_atomic(root / 'plan.json', dict(batch=batch, candidates=len(candidates), skipped=skipped, at=now()))
    return dict(batch=batch, candidates=len(candidates), skipped=skipped)


def environment(adapter):
    out = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), numpy=np.__version__,
               rig=adapter.rig.to_json(),
               conda_env=os.environ.get('CONDA_DEFAULT_ENV', Path(sys.executable).resolve().parent.parent.name))
    out['git_commit'] = {}
    for label, path in [('intervene', REPO_ROOT), ('workspace', WORKSPACE),
                        ('model', adapter.params['adapter'].get('root', Path.cwd()))]:
        try:
            out['git_commit'][label] = subprocess.check_output(
                ['git', '-C', str(path), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                universal_newlines=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            out['git_commit'][label] = None
    for name in ('torch', 'habitat', 'habitat_sim'):
        module = sys.modules.get(name)
        if module is not None:
            out[name] = getattr(module, '__version__', 'unversioned')
    return out


def valid_run(rec, params, cand, batch):
    store.validate(rec, store.RUN)
    if rec['batch'] != batch or rec['params'] != params or rec['cand'] != cand:
        raise ValueError('archived run does not match immutable batch/candidate')
    if not rec['trajectory']:
        raise ValueError('archived run has no trajectory')
    if not rec.get('checks', {}).get('start_reached'):
        raise ValueError('archived run did not verify intervention start')
    for side in ('orig', 'branch'):
        for metric in METRICS:
            value = rec[side].get(metric)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError('native %s metric missing/nonfinite: %s' % (side, metric))
    if params['arm'] != 'none' and not any(x['step'] == cand['step'] for x in rec['blackout']):
        raise ValueError('archived run lacks start blackout observation')
    return rec


ONE_SHOT = ('simplevln',)       # sampled generation: one attempt per candidate, repeated only on explicit request


def free_gpu():
    """After a failed candidate: collect the dead adapter's cycles and return cached GPU memory."""
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(model, batch, episodes=None, limit=None, fail_fast=False, retry_errors=False):
    params, candidates = load_batch(model, batch)
    if episodes:
        candidates = [c for c in candidates if c['ep'] in set(map(str, episodes))]
    adapter, finished, skipped, failed = None, 0, 0, 0
    last_error, repeats = None, 0
    with exclusive(batch_dir(model, batch) / '.run.lock'):
        try:
            for c in candidates:
                if limit is not None and finished + failed >= limit:
                    break
                path = store.run_path(model, c['split'], c['ep'], c['step'], batch)
                if path.exists():
                    valid_run(json.loads(path.read_text()), params, c, batch)
                    skipped += 1
                    continue
                attempt = path.parent / 'attempt.json'
                if attempt.exists() and model in ONE_SHOT and not retry_errors:
                    # SimpleVLN samples: never repeat an attempt silently. OmniNav and ETPNav are deterministic, so a
                    # candidate that failed or was interrupted is simply run again.
                    skipped += 1
                    print('SKIP already attempted %s (sampled model; --retry-errors repeats it)' % c['ep'], flush=True)
                    continue
                store.write_json_atomic(attempt, dict(at=now(), cand=c, status='started', pid=os.getpid()))
                start = time.time()
                try:
                    module = importlib.import_module('.agents.' + model, __package__)
                    if adapter is None:
                        adapter = getattr(module, ADAPTERS[model])(params, path.parent)
                    adapter.output = path.parent
                    result = adapter.run(c)
                    if adapter.rig.to_json() != params['rig']:
                        raise ValueError('live rig differs from planned native config')
                    orig = (adapter.original_metrics(c) if hasattr(adapter, 'original_metrics') else
                            original_metrics(primary(model, c['split'], c['ep'])))
                    record = dict(batch=batch, params=params, cand=c, orig=orig, seconds=time.time() - start,
                                  at=now(), env=environment(adapter))
                    record.update(result)
                    valid_run(record, params, c, batch)
                    store.write_json_atomic(path, record)
                    store.write_json_atomic(attempt, dict(at=now(), status='complete', cand=c))
                    finished += 1
                    last_error, repeats = None, 0
                    print('DONE %s %s: SR=%s SPL=%s nDTW=%s' %
                          (model, c['ep'], result['branch']['success'], result['branch']['spl'], result['branch']['ndtw']), flush=True)
                except Exception as exc:
                    failed += 1
                    store.append_error(model, batch, dict(ep=c['ep'], step=c['step'], at=now(),
                                       error=repr(exc), error_type=type(exc).__name__, traceback=traceback.format_exc()))
                    store.write_json_atomic(attempt, dict(at=now(), status='error', cand=c, error=repr(exc)))
                    # Keep the adapter across failures. It owns the loaded policy and the native environment;
                    # discarding it reloaded the model for every following candidate, and a reload while the GPU
                    # was still holding the previous copy sent layers to CPU (device_map="auto") before the
                    # native .cuda(), which then raised and failed every later candidate in turn.
                    if fail_fast:
                        raise
                    # One failure can leave every later candidate failing the same way (an environment that
                    # cannot be reset); stop instead of marking the whole queue as failed.
                    key = (type(exc).__name__, str(exc)[:120])
                    repeats = repeats + 1 if key == last_error else 1
                    last_error = key
                    if repeats >= 2:
                        raise RuntimeError('stopped after %d consecutive identical errors (%s: %s); finished %d, '
                                           'failed %d. The failed candidates rerun on the next run.' %
                                           (repeats, key[0], key[1], finished, failed))
                free_gpu()
        finally:
            if adapter is not None:
                adapter.close()
    return dict(batch=batch, complete=finished, skipped=skipped, errors=failed)


def runs(model, batch):
    params, candidates = load_batch(model, batch)
    records, missing = [], []
    for c in candidates:
        path = store.run_path(model, c['split'], c['ep'], c['step'], batch)
        if not path.exists():
            missing.append(c['ep'])
            continue
        rec = valid_run(json.loads(path.read_text()), params, c, batch)
        records.append(rec)
    return params, records, missing


def analyze(model, batch, n_boot=1000, level=.95, seed=0):
    params, records, missing = runs(model, batch)
    errors = [e for e in store.read_jsonl(store.experiment_dir(model) / 'errors.jsonl') if e.get('batch') == batch]
    _, candidates = load_batch(model, batch)
    attempts = Counter()
    for c in candidates:
        path = store.run_path(model, c['split'], c['ep'], c['step'], batch).parent / 'attempt.json'
        if path.is_file():
            attempts[json.loads(path.read_text())['status']] += 1
    weights = stats.cluster_weights([r['cand']['trajectory_id'] for r in records], n_boot, seed)
    groups = {'all': np.ones(len(records), bool)}
    for field in ('category', 'status', 'scene'):
        values = [r['cand'][field] for r in records]
        for value in sorted(set(values)):
            groups[field + ':' + value] = np.asarray([x == value for x in values])
    result = dict(batch=batch, params=params, n_boot=int(n_boot), level=float(level), seed=int(seed),
                  checks=dict(complete=len(records), missing=missing, errors=len(errors),
                              error_types=dict(Counter(e.get('error_type', e['error'].split('(')[0]) for e in errors)),
                              attempts=dict(attempts),
                              truncated=sum(bool(r.get('termination', {}).get('truncated')) for r in records),
                              clusters=len(set(r['cand']['trajectory_id'] for r in records)),
                              native_metric_missing={side: {m: sum(r[side].get(m) is None for r in records)
                                                     for m in METRICS} for side in ('orig', 'branch')}),
                  groups=stats.paired(records, METRICS, weights, groups, level=level),
                  flips=stats.flips(records))
    store.validate(result, store.SUMMARY)
    root = RESULTS / model / batch
    store.write_json_atomic(root / 'summary.json', result)
    return result
