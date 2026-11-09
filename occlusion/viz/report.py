"""Static report/review pages; all raw artifacts remain in the formal archive."""
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .. import runner
from ..core import store
from ..paths import RESULTS
from ..select.common import primary

STYLE = '''body{font:15px system-ui,sans-serif;background:#f4f6fa;color:#17233b;margin:28px auto;max-width:1260px;padding:0 18px}h1,h2{color:#142f57}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;border-bottom:1px solid #dde3ec}th:first-child,td:first-child{text-align:left}th{background:#e6edf7}section,article{background:white;border:1px solid #dce2eb;border-radius:9px;padding:20px;margin:18px 0}.metrics{display:flex;gap:24px;flex-wrap:wrap}.muted{color:#62728a}.plot{max-width:100%;height:auto}.views{display:flex;flex-wrap:wrap;gap:12px}.view{max-width:390px}img{max-width:100%;height:auto}a{color:#225dab}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f9;padding:12px}input,select{padding:8px;margin-right:12px}summary{cursor:pointer}.legend span{margin-right:18px}.ok{color:#17764b}.bad{color:#bd3434}'''


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    return '—' if value is None else '%.4f' % float(value)


def interval(value):
    mean, lo, hi, n = value
    return '%s [%s, %s] (n=%s)' % (number(mean), number(lo), number(hi), n)


def link(path, page_dir):
    return quote(os.path.relpath(str(path), str(page_dir)).replace(os.sep, '/'), safe='/')


def document(title, body, script=''):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>%s</title><style>%s</style><body>%s%s</body></html>' %
            (esc(title), STYLE, body, script))


def topdown(original, branch, blackout):
    """A world x/z view, not a navigability map or fabricated reference path."""
    series = []
    for rows, color in ((original, '#71849f'), (branch, '#125fc0'), (blackout, '#dc6b24')):
        points = [store._position(r) for r in rows]
        points = np.asarray([p for p in points if p is not None], np.float64).reshape(-1, 3)
        if len(points):
            if not np.isfinite(points).all():
                raise ValueError('nonfinite trajectory position')
            series.append((points[:, [0, 2]], color))
    if not series:
        return '<p>没有可用轨迹。</p>'
    all_points = np.concatenate([p for p, _ in series])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    center = (lo + hi) / 2.
    scale = 390. / max(float((hi - lo).max()), 1.)
    def xy(p):
        return (p - center) * scale + [250., 235.]
    parts = ['<svg class="plot" viewBox="0 0 500 475" role="img" aria-label="原轨迹与分支俯视对比">',
             '<rect width="500" height="475" fill="#f8fafc"/>']
    for points, color in series:
        transformed = xy(points)
        coords = ' '.join('%.2f,%.2f' % (p[0], p[1]) for p in transformed)
        parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.5"/>' % (coords, color))
        if color == '#dc6b24':
            for p in transformed:
                parts.append('<circle cx="%.2f" cy="%.2f" r="3.5" fill="%s"/>' % (p[0], p[1], color))
    parts.append('<text x="15" y="20" fill="#53657d">世界 x 向右，z 向下；非地图</text>')
    bar = min(3., 140. / scale)
    parts.append('<path d="M20 447h%.2f" stroke="#17233b" stroke-width="3"/><text x="20" y="467">%.2f m</text></svg>' %
                 (bar * scale, bar))
    return ''.join(parts)


def image_gallery(record, source_dir, page_dir):
    parts = []
    for row in record.get('images', []):
        parts.append('<details><summary>观测 / 规划步 %s</summary><div class="views">' % esc(row['step']))
        for view, images in sorted(row['views'].items()):
            parts.append('<div class="view"><h4>%s</h4>' % esc(view))
            for kind in ('clean', 'black', 'mask'):
                if kind not in images:
                    continue
                path = source_dir / images[kind]
                if not path.is_file():
                    raise FileNotFoundError('referenced image is missing: %s' % path)
                parts.append('<figure><figcaption>%s</figcaption><img loading="lazy" src="%s" alt="%s %s"></figure>' %
                             (kind, esc(link(path, page_dir)), esc(view), kind))
            parts.append('</div>')
        parts.append('</div></details>')
    return ''.join(parts) or '<p class="muted">本记录未保存涂黑图；可用 offline 生成原轨迹 mask 复核图。</p>'


def build(model, batch):
    params, records, missing = runner.runs(model, batch)
    root = RESULTS / model / batch
    summary_path = root / 'summary.json'
    if not summary_path.exists():
        runner.analyze(model, batch)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    store.validate(summary, store.SUMMARY)
    if summary['params'] != params or summary['checks']['complete'] != len(records) or summary['checks']['missing'] != missing:
        raise ValueError('summary is stale; rerun analyze before viz')
    title = '%s / %s' % (model, batch)
    body = '<h1>%s</h1><p><a href="review.html">逐条分支复核 →</a> · <a href="summary.json">summary.json</a></p>' % esc(title)
    checks = summary['checks']
    body += '<section><h2>完成与检查</h2><pre>%s</pre></section>' % esc(json.dumps(checks, ensure_ascii=False, indent=2))
    body += '<section><h2>成功翻转</h2><div class="metrics">'
    for name, count in summary['flips'].items():
        body += '<div><strong>%s</strong><p>%s</p></div>' % (esc(name), count)
    body += '</div></section><section><h2>成簇配对 Bootstrap</h2><p>按 trajectory_id 成簇，%s 次，置信水平 %.1f%%。差值 = 分支 − 原轨迹。</p>' % (summary['n_boot'], summary['level'] * 100.)
    body += '<table><thead><tr><th>组 / 指标</th><th>原轨迹 [CI]</th><th>分支 [CI]</th><th>配对差 [CI]</th></tr></thead><tbody>'
    for group, values in summary['groups'].items():
        for metric in runner.METRICS:
            v = values[metric]
            body += '<tr><td>%s / %s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % tuple(map(esc, (group, metric, interval(v['orig']), interval(v['branch']), interval(v['delta']))))
    body += '</tbody></table></section><section><h2>完整批次参数</h2><pre>%s</pre></section>' % esc(json.dumps(params, ensure_ascii=False, indent=2))
    store._atomic_bytes(root / 'index.html', document(title, body).encode('utf-8'))
    review = '<h1>分支复核 · %s</h1><p><a href="index.html">← 分组报告</a></p>' % esc(title)
    review += '<input id="search" placeholder="筛选 episode / 类别 / 场景"><span class="muted">灰：原轨迹 · 蓝：分支 · 橙：涂黑观测位姿</span>'
    for record in records:
        cand = record['cand']
        source = store.run_path(model, cand['split'], cand['ep'], cand['step'], batch).parent
        label = '%s | %s | %s | %s' % (cand['ep'], cand['category'], cand['scene'], cand['status'])
        review += '<article data-search="%s"><h2>%s</h2><p>d=%s (%s)，offset=%s，start=%s，center=%s°</p>' % tuple(map(esc, (label.lower(), label, cand['d'], cand['d_unit'], cand['offset'], cand['step'], cand['center'])))
        review += '<p><a href="%s">原始 run.json</a></p><div class="metrics">' % esc(link(source / 'run.json', root))
        for metric in runner.METRICS:
            review += '<span><strong>%s</strong> %s → %s</span>' % (metric, number(record['orig'].get(metric)), number(record['branch'].get(metric)))
        review += '</div>' + topdown(primary(model, cand['split'], cand['ep']),
                                    [dict(position=p) for p in record['positions']] if model == 'etpnav' else record['trajectory'],
                                    record['blackout'])
        review += '<h3>闭环分支观测</h3>' + image_gallery(record, source, root)
        offline_dir = runner.batch_dir(model, batch) / 'offline' / cand['ep']
        if (offline_dir / 'render.json').is_file():
            offline = json.loads((offline_dir / 'render.json').read_text())
            review += '<h3>原轨迹离线渲染（非闭环分支）</h3>' + image_gallery(offline, offline_dir, root)
        review += '<details><summary>核对与遮挡统计</summary><pre>%s</pre></details></article>' % esc(json.dumps(dict(checks=record.get('checks', {}), blackout=record['blackout']), ensure_ascii=False, indent=2))
    if missing:
        review += '<section><h2>尚无有效 run.json</h2><p>%s</p></section>' % esc(', '.join(missing))
    script = '<script>document.getElementById("search").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("article[data-search]").forEach(x=>{x.hidden=!x.dataset.search.includes(q)})});</script>'
    store._atomic_bytes(root / 'review.html', document('复核 · ' + title, review, script).encode('utf-8'))
    return dict(batch=batch, complete=len(records), missing=missing, report=str(root / 'index.html'), review=str(root / 'review.html'))
