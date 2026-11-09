"""Region from a semantic model behind an API.

Three replaceable parts: a request builder (frame -> API input), the Responses-API transport, and a response
parser (text -> region spec). The parser returns a heading/columns spec, so the existing validation and tracking
apply unchanged. Pillow is imported lazily, only when images are encoded.
"""

import base64
import hashlib
import http.client
import io
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ..paths import ENV_FILE
from .geom import wrap180

USER_AGENT = 'occlusion-client/0.1'  # the gateway's Cloudflare rejects urllib's default agent (HTTP 403, 1010)
# Seconds to wait before each retry of a transient failure (connection reset, timeout, HTTP 429/5xx). No literature
# basis; a retry does not change the request, so the model sees the same input.
RETRY_WAITS = (5, 15, 45)
COMMON = ('base_url', 'api_key_env', 'model_name', 'reasoning_effort', 'request', 'parser', 'timeout')


def load_env(path=None):
    """KEY=VALUE lines of .env into os.environ; variables already set in the shell win."""
    path = Path(path or ENV_FILE)
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


PROVIDER_FILE = Path(__file__).resolve().parents[1] / 'configs' / 'provider.json'
PROVIDER_FIELDS = ('base_url', 'api_key_env', 'model_name')


def provider_options(options):
    """Options with the endpoint taken from configs/provider.json when that file exists.

    The batch parameters (and so the batch name and every archived record) keep the provider they were planned
    with; switching provider only needs this file. Each region record states the provider actually used.
    """
    if not PROVIDER_FILE.is_file():
        return options
    override = json.loads(PROVIDER_FILE.read_text(encoding='utf-8'))
    unknown = set(override) - set(PROVIDER_FIELDS)
    if unknown or not all(isinstance(v, str) and v.strip() for v in override.values()):
        raise ValueError('%s may only set non-empty strings for %s' % (PROVIDER_FILE, PROVIDER_FIELDS))
    return dict(options, **override)


def api_key(options):
    options = provider_options(options)
    load_env()
    name = options['api_key_env']
    value = os.environ.get(name, '')
    if not value:
        raise RuntimeError('environment variable %s is not set (put it in .env or the shell)' % name)
    return value


def _png(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def build_views_input(options, cand, frame, rig):
    """The prompt, then each requested view as a 'view=<name> hfov=<deg> center=<deg>' line and its image.

    Only two candidate fields are read: the planned direction (cand['center'] relative to the robot's heading, in
    degrees, positive to the left) and the episode instruction (cand['extra']['instruction']). Annotations,
    categories, deviation steps and ground-truth paths are not sent. hfov is the view's full horizontal field of
    view; center is the direction of its image centre, in the same robot-relative convention. The prompt must
    contain {direction} and {instruction}.
    """
    if cand.get('center') is None:
        raise ValueError('candidate has no planned direction (center)')
    direction = round(float(wrap180(float(cand['center']) - float(frame.pose.yaw))), 1)
    text = (cand.get('extra') or {}).get('instruction')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('candidate has no instruction (re-plan the batch)')
    values = {'direction': '%.1f' % direction, 'instruction': text.strip()}
    prompt = re.sub(r'\{(direction|instruction)\}', lambda m: values[m.group(1)], options['prompt'])
    content = [{'type': 'input_text', 'text': prompt}]
    info = {'prompt': prompt, 'direction': direction, 'instruction': values['instruction'], 'views': {}}
    for name in options['views']:
        view = rig.view(name)
        arr = np.asarray(frame.rgb[name])
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError('view %s: rgb must be uint8 [H, W, 3]' % name)
        if arr.shape[1] != view.mask_hw[1]:
            raise ValueError('view %s: image width %d != mask width %d' % (name, arr.shape[1], view.mask_hw[1]))
        png = _png(arr)
        label = 'view=%s hfov=%.1f center=%.1f' % (name, view.hfov, view.yaw_offset)
        content.append({'type': 'input_text', 'text': label})
        content.append({'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(png).decode()})
        info['views'][name] = {'label': label, 'shape': list(arr.shape), 'png_sha256': hashlib.sha256(png).hexdigest()}
    return [{'role': 'user', 'content': content}], info


def call_responses(options, payload):
    """POST {base_url}/v1/responses; returns (text of the message output, metadata). Errors never carry the key."""
    options = provider_options(options)
    body = {'model': options['model_name'], 'input': payload,
            'reasoning': {'effort': options['reasoning_effort']}}
    req = urllib.request.Request(
        options['base_url'].rstrip('/') + '/v1/responses', data=json.dumps(body).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + api_key(options), 'Content-Type': 'application/json',
                 'User-Agent': USER_AGENT})
    data, attempts = None, 0
    for wait in (0,) + tuple(RETRY_WAITS):
        if wait:
            time.sleep(wait)
        attempts += 1
        try:
            data = json.loads(urllib.request.urlopen(req, timeout=float(options['timeout'])).read().decode('utf-8'))
            break
        except urllib.error.HTTPError as e:
            detail = 'region model HTTP %d: %s' % (e.code, e.read()[:500].decode('utf-8', 'replace'))
            if not (e.code == 429 or e.code >= 500):
                raise RuntimeError(detail)
            failure = detail
        except (urllib.error.URLError, socket.timeout, ConnectionError, http.client.HTTPException) as e:
            failure = 'region model request failed: %s' % (getattr(e, 'reason', None) or repr(e),)
    if data is None:
        raise RuntimeError('%s (after %d attempts)' % (failure, attempts))
    if data.get('status') != 'completed':
        raise RuntimeError('region model response not completed: %r' % (data.get('status'),))
    # The gateway once answered a gpt-5.6-luna request with gpt-6.1-sol, so a silent reroute is an error.
    model = str(data.get('model', ''))
    if not model.startswith(options['model_name']):
        raise RuntimeError('region model answered by %r, requested %r' % (model, options['model_name']))
    effort = (data.get('reasoning') or {}).get('effort')
    if effort is not None and effort != options['reasoning_effort']:
        raise RuntimeError('region model reasoning effort %r, requested %r' % (effort, options['reasoning_effort']))
    text = ''.join(c.get('text', '') for o in data.get('output', []) if o.get('type') == 'message'
                   for c in o.get('content', []) if c.get('type') == 'output_text')
    if not text:
        raise RuntimeError('region model returned no text')
    return text, {'model': model, 'effort': effort, 'usage': data.get('usage'), 'attempts': attempts,
                  'base_url': options['base_url']}


def parse_x_range(text, options, rig):
    """'view=<name> x_range=[a, b]' (a, b in 0..1 across the image) -> columns spec of the mask width."""
    ranges = re.findall(r'x_range\s*=\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]', text)
    if len(ranges) != 1:
        raise ValueError('expected exactly one x_range=[a, b] in the model output, found %d: %r' %
                         (len(ranges), text[:300]))
    names = re.findall(r'view\s*=\s*([A-Za-z0-9_.\-]+)', text)
    if len(set(names)) > 1:
        raise ValueError('model output names several views: %r' % sorted(set(names)))
    if names:
        view = names[0]
    elif len(options['views']) == 1:
        view = options['views'][0]
    else:
        raise ValueError('model output names no view and several were sent: %r' % (text[:300],))
    if view not in options['views']:
        raise ValueError('model output view %r was not among the views sent %r' % (view, options['views']))
    x0, x1 = sorted(float(v) for v in ranges[0])
    if not (0.0 <= x0 and x1 <= 1.0):
        raise ValueError('x_range must lie in [0, 1], got [%g, %g]' % (x0, x1))
    width = rig.view(view).mask_hw[1]
    first = min(int(math.floor(x0 * width)), width - 1)
    last = max(min(int(math.ceil(x1 * width)) - 1, width - 1), first)
    return {'kind': 'columns', 'view': view, 'bounds': [first, last]}


REQUESTS = {'views': (build_views_input, ('views', 'prompt'))}
PARSERS = {'x_range': parse_x_range}


def check_options(options):
    """Shape and type check of region_spec.options (no network, no environment)."""
    if not isinstance(options, dict) or options.get('request') not in REQUESTS or options.get('parser') not in PARSERS:
        raise ValueError('model options need request in %s and parser in %s' % (sorted(REQUESTS), sorted(PARSERS)))
    extras = REQUESTS[options['request']][1]
    if set(options) != set(COMMON) | set(extras):
        raise ValueError('model options have missing or unexpected fields: %s' % sorted(set(options) ^ (set(COMMON) | set(extras))))
    for key in ('base_url', 'api_key_env', 'model_name', 'reasoning_effort', 'prompt'):
        if key in options and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError('model option %s must be a non-empty string' % key)
    if options['request'] == 'views' and not all(k in options['prompt'] for k in ('{direction}', '{instruction}')):
        raise ValueError('model option prompt must contain {direction} and {instruction}')
    if not options['base_url'].startswith(('http://', 'https://')):
        raise ValueError('model option base_url must start with http:// or https://')
    timeout = options['timeout']
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
        raise ValueError('model option timeout must be a positive number of seconds')
    if 'views' in options and (not isinstance(options['views'], list) or not options['views'] or
                               any(not isinstance(v, str) or not v for v in options['views'])):
        raise ValueError('model option views must be a non-empty list of view names')


def check_rig(options, rig):
    """Views exist in the rig, and the API key variable is available."""
    for name in options.get('views', []):
        try:
            rig.view(name)
        except KeyError:
            raise ValueError('model option views: unknown view %r (rig has %s)' % (name, [v.name for v in rig.views]))
    api_key(options)


def region_from_model(options, cand, frame, rig):
    """(columns spec, record) for the first active frame."""
    build, _ = REQUESTS[options['request']]
    payload, request = build(options, cand, frame, rig)
    text, meta = call_responses(options, payload)
    spec = PARSERS[options['parser']](text, options, rig)
    record = {'request': request, 'response_text': text, 'spec': spec}
    record.update(meta)
    return spec, record
