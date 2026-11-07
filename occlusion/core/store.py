"""File contracts and archive I/O for occlusion intervention runs."""

import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ARCHIVE = Path('/media/jwm/大数据/result')
SERIES = {'omninav': 'C6002o', 'simplevln': 'C6002s', 'etpnav': 'C6002e'}
D_UNIT = {'omninav': 'forward_step', 'simplevln': 'waypoint', 'etpnav': 'planning'}

CANDIDATE = {
    'model': str, 'split': str, 'ep': str, 'scene': str, 'trajectory_id': str,
    'd': int, 'd_unit': str, 'offset': int, 'step': int,
    'center': (float, type(None)), 'category': str, 'status': str,
    'review_file': str, 'source_sha256': str, 'rule_version': str, 'extra': dict,
}
RUN = {
    'batch': str, 'params': dict, 'cand': dict, 'orig': dict, 'branch': dict,
    'blackout': list, 'trajectory': list, 'seconds': (int, float), 'at': str, 'env': dict,
}
SUMMARY = {
    'batch': str, 'params': dict, 'n_boot': int, 'level': (int, float), 'seed': int,
    'checks': dict, 'groups': dict, 'flips': dict,
}


def _json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                        allow_nan=False) + '\n').encode('utf-8')


def _hash(path):
    digest = hashlib.sha256()
    size = 0
    with open(str(path), 'rb') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def _under_archive(path):
    try:
        return os.path.commonpath([str(ARCHIVE.resolve()), str(Path(path).resolve())]) == str(ARCHIVE.resolve())
    except (AttributeError, ValueError):
        root = str(ARCHIVE.resolve()) + os.sep
        return str(Path(path).resolve()).startswith(root)


def _archive_available(metadata=False):
    if not ARCHIVE.is_dir():
        raise FileNotFoundError('archive root is unavailable: %s' % ARCHIVE)
    if metadata:
        required = ('checksums.jsonl', 'manifest.json', 'README.md', 'settings.json')
        missing = [name for name in required if not (ARCHIVE / name).is_file()]
        if missing:
            raise FileNotFoundError('archive metadata is incomplete at %s: %s' %
                                    (ARCHIVE, ', '.join(missing)))
    return ARCHIVE


def _component(value, label):
    value = str(value)
    if not value or value in ('.', '..') or '/' in value or '\\' in value or '\x00' in value:
        raise ValueError('invalid %s path component: %r' % (label, value))
    return value


def validate(rec, schema):
    if not isinstance(rec, dict):
        raise ValueError('record must be a dict')
    for key, expected in schema.items():
        if key not in rec:
            raise ValueError('record is missing required field %r' % key)
        value = rec[key]
        allowed = expected if isinstance(expected, tuple) else (expected,)
        ok = False
        for typ in allowed:
            if typ is type(None):
                ok |= value is None
            elif typ is float:
                ok |= isinstance(value, (int, float)) and not isinstance(value, bool)
            elif typ is int:
                ok |= isinstance(value, int) and not isinstance(value, bool)
            else:
                ok |= isinstance(value, typ)
        if not ok:
            names = ', '.join('None' if t is type(None) else t.__name__ for t in allowed)
            raise ValueError('field %r must be %s, got %s' % (key, names, type(value).__name__))
    return None


def batch_name(params):
    if not isinstance(params, dict):
        raise TypeError('params must be a dict')
    region = params.get('region_spec', params.get('region', {}))
    tracker = params.get('tracker_spec', params.get('tracker', {}))
    region_name = region.get('kind', 'region') if isinstance(region, dict) else str(region)
    tracker_name = tracker.get('kind', 'tracker') if isinstance(tracker, dict) else str(tracker)
    follow = params.get('follow')
    follow_name = 'all' if follow is None else str(int(follow))
    digest = hashlib.sha256(_json_bytes(params).rstrip(b'\n')).hexdigest()[:8]
    slug = lambda x: re.sub(r'[^A-Za-z0-9._+-]+', '-', str(x)).strip('-') or 'x'
    return '%s-%s-%s-%s' % (slug(region_name), slug(tracker_name), slug(follow_name), digest)


def _episode_path(split, ep):
    split = _component(split, 'split')
    ep = _component(ep, 'episode id')
    return _archive_available() / split / (split + '-' + ep)


def episode_dir(split, ep):
    return _episode_path(split, ep)


def run_path(model, split, ep, step, batch):
    model = _component(model, 'model')
    if model not in SERIES:
        raise ValueError('unknown model: %s' % model)
    step = int(step)
    if step < 1 or step > 9999:
        raise ValueError('step must be in [1, 9999]')
    return (_episode_path(split, ep) / model / 'intervene' / SERIES[model] /
            ('%04d' % step) / _component(batch, 'batch') / 'run.json')


def experiment_dir(model):
    model = _component(model, 'model')
    if model not in SERIES:
        raise ValueError('unknown model: %s' % model)
    return _archive_available() / 'experiments' / model / 'intervene' / SERIES[model]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(str(path), 'r', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError('%s:%d is not valid JSON: %s' % (path, number, exc))
            if not isinstance(value, dict):
                raise ValueError('%s:%d must contain a JSON object' % (path, number))
            records.append(value)
    return records


def _append_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(fd, 'ab', closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def append_jsonl(path, rec):
    path = Path(path)
    before = _hash(path) if path.exists() else None
    _append_bytes(path, _json_bytes(rec))
    if _under_archive(path) and path.name != 'records.jsonl':
        record(_model_for_path(path), 'append', path, before)


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, str(path))
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_atomic(path, obj):
    path = Path(path)
    before = _hash(path) if path.exists() else None
    _atomic_bytes(path, _json_bytes(obj))
    if _under_archive(path):
        record(_model_for_path(path), 'rewrite' if before is not None else 'write', path, before)


def _model_for_path(path):
    parts = Path(path).resolve().parts
    for model in SERIES:
        if model in parts:
            return model
    return 'omninav'


def load_reviews(path, sha_of):
    path = Path(path)
    with open(str(path), 'r', encoding='utf-8') as stream:
        data = json.load(stream)
    reviews = data.get('reviews', data) if isinstance(data, dict) else data
    if isinstance(reviews, dict):
        iterable = reviews.values()
    elif isinstance(reviews, list):
        iterable = reviews
    else:
        raise ValueError('review file must contain a reviews list or mapping')
    out = {}
    for review in iterable:
        ep = str(review.get('episode_id', review.get('ep', '')))
        if not ep:
            raise ValueError('review is missing episode_id')
        if ep in out:
            raise ValueError('duplicate review for episode %s' % ep)
        expected = review.get('source_sha256')
        actual = sha_of(ep)
        if expected != actual:
            raise ValueError('source_sha256 mismatch for episode %s: review=%s current=%s' %
                             (ep, expected, actual))
        out[ep] = dict(review)
    return out


def _rollout_path(model, split, ep):
    return _episode_path(split, ep) / model / 'primary' / 'rollout.jsonl'


def append_deviation(split, ep, model, row):
    if model not in SERIES:
        raise ValueError('unknown model: %s' % model)
    path = _rollout_path(model, split, ep)
    if not path.is_file():
        raise FileNotFoundError('primary rollout does not exist: %s' % path)
    raw = path.read_bytes()
    records = read_jsonl(path)
    if any(r.get('event') == 'deviation' for r in records):
        raise ValueError('rollout already has a deviation event: %s' % path)
    event = dict(row)
    event['event'] = 'deviation'
    event.setdefault('d_unit', D_UNIT[model])
    if not raw.endswith(b'\n'):
        raise ValueError('rollout has no trailing newline; refusing to alter existing bytes')
    before = _hash(path)
    _append_bytes(path, _json_bytes(event))
    record(model, 'append', path, before)


def update_deviation(split, ep, model, fields):
    if model not in SERIES:
        raise ValueError('unknown model: %s' % model)
    path = _rollout_path(model, split, ep)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    found = []
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        rec = json.loads(line.decode('utf-8'))
        if rec.get('event') == 'deviation':
            found.append(len(new_lines))
            rec.update(fields)
            ending = b'\n' if line.endswith(b'\n') else b''
            new_lines.append(_json_bytes(rec).rstrip(b'\n') + ending)
        else:
            new_lines.append(line)
    if len(found) != 1:
        raise ValueError('expected one deviation event in %s, found %d' % (path, len(found)))
    before = _hash(path)
    _atomic_bytes(path, b''.join(new_lines))
    record(model, 'rewrite', path, before)


def _position(row):
    for key in ('position', 'pos'):
        if key in row and row[key] is not None:
            p = row[key]
            if isinstance(p, dict):
                p = [p[k] for k in ('x', 'y', 'z')]
            if len(p) == 3:
                return [float(x) for x in p]
    return None


def trajectory_from(model, split, ep, step):
    import numpy as np

    if model not in SERIES:
        raise ValueError('unknown model: %s' % model)
    step = int(step)
    if step < 1:
        raise ValueError('step must be positive')
    rows = read_jsonl(_rollout_path(model, split, ep))
    if model != 'etpnav':
        positions = []
        ready = next((r for r in rows if r.get('event') == 'ready'), None)
        if ready is not None:
            p = _position(ready)
            if p is not None:
                positions.append(p)
        for row in rows:
            if row.get('event') == 'step':
                p = _position(row)
                if p is not None:
                    positions.append(p)
        if step > len(positions):
            raise ValueError('step %d exceeds rollout positions (%d)' % (step, len(positions)))
        return np.asarray(positions[step - 1:], dtype=np.float64)

    positions = []
    position_rows = []
    for row in rows:
        if row.get('event') in ('position', 'step', 'done', 'ready'):
            p = _position(row)
            if p is not None:
                positions.append(p)
                position_rows.append(row)
    decisions = [r for r in rows if r.get('event') == 'decision']
    decision = next((r for i, r in enumerate(decisions, 1)
                     if int(r.get('planning_step', r.get('planning', r.get('plan_step', i)))) == step), None)
    if decision is None:
        raise ValueError('ETPNav rollout has no decision for planning step %d' % step)
    index = None
    for key in ('position_record', 'position_index', 'record_index', 'position_step', 'step_index'):
        if key in decision:
            index = int(decision[key])
            if decision.get(key + '_base') == 1:
                index -= 1
            break
    if index is None:
        # Native ETP end is the one-based position record reached by the
        # preceding decision. Repeated positions must NOT be matched by value.
        j = decisions.index(decision)
        index = 0 if j == 0 else int(decisions[j - 1]['end']) - 1
    if index is None or not 0 <= index < len(positions):
        raise ValueError('ETPNav decision %d lacks a valid position-record mapping' % step)
    return np.asarray(positions[index:], dtype=np.float64)


def done_keys(model, split, batch):
    model = _component(model, 'model')
    if model not in SERIES:
        raise ValueError('unknown model: %s' % model)
    root = _archive_available() / _component(split, 'split')
    done = set()
    for ep_path in root.glob(split + '-*'):
        base = ep_path / model / 'intervene' / SERIES[model]
        if not base.is_dir():
            continue
        for run in base.glob('*/%s/run.json' % _component(batch, 'batch')):
            try:
                rec = json.loads(run.read_text(encoding='utf-8'))
            except ValueError:
                continue
            try:
                validate(rec, RUN)
                validate(rec['cand'], CANDIDATE)
            except ValueError:
                continue
            if rec.get('batch') == batch:
                done.add((ep_path.name[len(split) + 1:], int(rec['cand']['step'])))
    return done


def append_error(model, batch, rec):
    path = experiment_dir(model) / 'errors.jsonl'
    event = dict(rec)
    event.setdefault('batch', batch)
    before = _hash(path) if path.exists() else None
    _append_bytes(path, _json_bytes(event))
    record(model, 'append', path, before)


def record(model, op, path, before=None):
    if op not in ('write', 'append', 'rewrite'):
        raise ValueError('unknown archive operation: %s' % op)
    path = Path(path)
    if not _under_archive(path) or not path.is_file():
        raise ValueError('record path must be an existing file inside archive: %s' % path)
    root = _archive_available()
    current = _hash(path)
    entry = {'op': op, 'path': path.resolve().relative_to(root.resolve()).as_posix(),
             'bytes': current['bytes'], 'sha256': current['sha256'],
             'before': before, 'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    exp_model = model if model in SERIES else _model_for_path(path)
    rec_path = experiment_dir(exp_model) / 'records.jsonl'
    _append_bytes(rec_path, _json_bytes(entry))


def _read_records(path):
    if not path.exists():
        return []
    return read_jsonl(path)


def _write_json_no_record(path, obj):
    _atomic_bytes(path, _json_bytes(obj))


def _checksum_map(path):
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        if not all(k in row for k in ('path', 'bytes', 'sha256')):
            raise ValueError('invalid checksum row in %s' % path)
        out[row['path']] = row
    return out


def _checksum_tree_entry(path, root):
    info = _hash(path)
    return {'path': Path(path).resolve().relative_to(root.resolve()).as_posix(),
            'bytes': info['bytes'], 'sha256': info['sha256']}


def _update_manifest(manifest, records, maint_rel, date):
    episodes = manifest.setdefault('episodes', [])
    if not isinstance(episodes, list):
        raise ValueError('manifest episodes must be a list')
    touched = {}
    for rec in records:
        rel = rec['path'].split('/')
        if len(rel) < 3 or not rel[0].startswith(('train', 'val')):
            continue
        split, ep_dir = rel[0], rel[1]
        if not ep_dir.startswith(split + '-'):
            continue
        ep = ep_dir[len(split) + 1:]
        key = (split, ep)
        touched.setdefault(key, set())
        if len(rel) > 2 and rel[2] in SERIES:
            touched[key].add('/'.join(rel[2:]))
    for (split, ep), files in touched.items():
        entry = None
        for item in episodes:
            item_split = item.get('split')
            item_ep = item.get('episode_id', item.get('id', item.get('episode')))
            item_path = item.get('directory', item.get('path', ''))
            if ((str(item_split) == split and str(item_ep) == ep) or
                    str(item_path).strip('/') == split + '/' + split + '-' + ep):
                entry = item
                break
        if entry is None:
            entry = {'directory': split + '/' + split + '-' + ep, 'files': []}
            episodes.append(entry)
        current = entry.setdefault('files', [])
        if not isinstance(current, list):
            raise ValueError('manifest episode files must be a list')
        for name in sorted(files):
            if name not in current:
                current.append(name)
        current.sort()
    counts = manifest.setdefault('counts', {})
    for model, series in SERIES.items():
        prefix = 'alternative run.json：%s/intervene/%s' % (model, series)
        counts.pop('alternative run.json: %s/intervene/%s' % (model, series), None)
        start = '%s/intervene/%s/' % (model, series)
        counts[prefix] = sum(1 for item in episodes for name in set(item.get('files', []))
                             if name.startswith(start) and name.endswith('/run.json'))
    experiments = manifest.setdefault('experiments', [])
    if isinstance(experiments, list):
        for rec in records:
            if rec['path'].startswith('experiments/') and rec['path'] not in experiments:
                experiments.append(rec['path'])
        experiments.sort()
    maintenance = manifest.setdefault('maintenance', [])
    if isinstance(maintenance, list):
        for name in ('checksums.before.jsonl', 'manifest.before.json', 'README.before.md',
                     'settings.before.json', 'records.jsonl'):
            value = maint_rel + '/' + name
            if value not in maintenance:
                maintenance.append(value)
        maintenance.sort()
    manifest['c6002_' + date.replace('-', '')] = {
        'date': date, 'records': maint_rel + '/records.jsonl',
        'backups': [maint_rel + '/' + name for name in
                    ('checksums.before.jsonl', 'manifest.before.json', 'README.before.md',
                     'settings.before.json')],
    }


def _update_settings(settings):
    fields = settings.setdefault('fields', {})
    if not isinstance(fields, dict):
        raise ValueError('settings fields must be an object')
    fields.setdefault('deviation', {
        'event': '人工标注的偏离点及其模型步号、来源校验和、换算后的涂黑步与区域中心',
        'd': '偏离点编号；单位由 d_unit 指定', 'd_unit': 'forward_step | waypoint | planning',
        'position_record': '仅 ETPNav：规划编号对应的位置记录序号',
        'center': '方位带世界朝向；ETPNav 在 none 批次后补写',
        'source_sha256': '产生人工标注时原始记录的 SHA-256',
    })
    experiments = settings.setdefault('experiments', {})
    if not isinstance(experiments, dict):
        raise ValueError('settings experiments must be an object')
    for model, series in SERIES.items():
        experiments.setdefault(series, {
            'path': 'experiments/%s/intervene/%s' % (model, series),
            'run_path': '%s/intervene/%s/<step>/<batch>/run.json' % (model, series),
            'fields': ['batch', 'params', 'cand', 'orig', 'branch', 'blackout',
                       'trajectory', 'seconds', 'at', 'env'],
        })


def _update_readme(path, date):
    text = path.read_text(encoding='utf-8')
    marker = '## C6002 偏离点遮挡干预'
    addition = ('\n\n' + marker + '\n\n' +
                'OmniNav、SimpleVLN、ETPNav 的偏离点遮挡分支分别存放于 ' +
                '`omninav/intervene/C6002o/`、`simplevln/intervene/C6002s/`、' +
                '`etpnav/intervene/C6002e/`。每条分支以 `run.json` 记录批次参数、原始与分支指标、' +
                '逐步遮挡日志及轨迹；模型级候选和维护记录存放于 `experiments/<model>/intervene/C6002<o|s|e>/`。\n')
    if marker not in text:
        text = text.rstrip() + addition
    history = '### %s — C6002 归档维护' % date
    if history not in text:
        text = text.rstrip() + ('\n\n' + history + '\n\n偏离点标注、候选与分支产物已入册；原始轨迹保留，元数据修改前备份与操作清单位于 `maintenance/c6002-%s/`。\n' % date.replace('-', ''))
    _atomic_bytes(path, text.encode('utf-8'))


def commit_meta(models):
    root = _archive_available(metadata=True)
    models = list(dict.fromkeys(models))
    bad = set(models) - set(SERIES)
    if bad:
        raise ValueError('unknown models: %s' % sorted(bad))
    now = datetime.datetime.now()
    date = now.strftime('%Y%m%d')
    date_iso = now.strftime('%Y-%m-%d')
    maint = root / 'maintenance' / ('c6002-' + date)
    if maint.exists():
        raise FileExistsError('maintenance directory already exists: %s' % maint)
    maint.mkdir(parents=True)
    source_names = ('checksums.jsonl', 'manifest.json', 'README.md', 'settings.json')
    backup_names = ('checksums.before.jsonl', 'manifest.before.json',
                    'README.before.md', 'settings.before.json')
    for src_name, dst_name in zip(source_names, backup_names):
        shutil.copy2(str(root / src_name), str(maint / dst_name))
    merged = {}
    for model in models:
        for row in _read_records(root / 'experiments' / model / 'intervene' /
                                 SERIES[model] / 'records.jsonl'):
            merged[row['path']] = row
    records = list(merged.values())
    records.sort(key=lambda x: x['path'])
    _atomic_bytes(maint / 'records.jsonl', b''.join(_json_bytes(row) for row in records))

    checksums = _checksum_map(root / 'checksums.jsonl')
    for row in records:
        target = root / row['path']
        if not target.is_file():
            raise FileNotFoundError('recorded archive file is missing: %s' % target)
        checksums[row['path']] = _checksum_tree_entry(target, root)

    manifest_path = root / 'manifest.json'
    settings_path = root / 'settings.json'
    readme_path = root / 'README.md'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    _update_manifest(manifest, records, maint.relative_to(root).as_posix(), date_iso)
    _update_settings(settings)
    _update_readme(readme_path, date_iso)

    # Include maintenance artifacts and experiment logs in the archive-wide checksum index.
    for path in maint.rglob('*'):
        if path.is_file():
            checksums[path.relative_to(root).as_posix()] = _checksum_tree_entry(path, root)
    for model in models:
        exp = root / 'experiments' / model / 'intervene' / SERIES[model]
        if exp.is_dir():
            for path in exp.rglob('*'):
                if path.is_file():
                    rel = path.relative_to(root).as_posix()
                    checksums[rel] = _checksum_tree_entry(path, root)
                    if rel not in manifest['experiments']:
                        manifest['experiments'].append(rel)
    manifest['experiments'].sort()
    _write_json_no_record(settings_path, settings)
    _write_json_no_record(manifest_path, manifest)
    checksums['settings.json'] = _checksum_tree_entry(settings_path, root)
    checksums['manifest.json'] = _checksum_tree_entry(manifest_path, root)
    checksums['README.md'] = _checksum_tree_entry(readme_path, root)
    rows = [checksums[key] for key in sorted(checksums) if key != 'checksums.jsonl']
    _atomic_bytes(root / 'checksums.jsonl', b''.join(_json_bytes(row) for row in rows))


def append_error_record(model, batch, rec):
    """Backward-compatible descriptive alias for append_error."""
    append_error(model, batch, rec)
