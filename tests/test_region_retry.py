"""Retry rules of the region model transport, with urlopen replaced: no network, no key, no waiting."""
import io
import json
import urllib.error

import pytest

from occlusion.core import semantic

OPTIONS = {'base_url': 'http://gateway.invalid', 'api_key_env': 'X', 'model_name': 'm', 'reasoning_effort': 'high',
           'timeout': 5}
GOOD = {'status': 'completed', 'model': 'm-2026', 'reasoning': {'effort': 'high'}, 'usage': {},
        'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'view=front x_range=[0.1, 0.2]'}]}]}


class Reply(object):
    def read(self):
        return json.dumps(GOOD).encode('utf-8')


def http_error(code):
    return urllib.error.HTTPError('http://gateway.invalid', code, 'x', {}, io.BytesIO(b'<html>%d</html>' % code))


@pytest.fixture
def transport(monkeypatch):
    state = {'calls': 0, 'script': [], 'waits': []}

    def urlopen(req, timeout=None):
        step = state['script'][state['calls']]
        state['calls'] += 1
        if isinstance(step, Exception):
            raise step
        return Reply()
    monkeypatch.setattr(semantic.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(semantic, 'provider_options', lambda o: o)
    monkeypatch.setattr(semantic, 'api_key', lambda o: 'not-a-key')
    monkeypatch.setattr(semantic.time, 'sleep', lambda s: state['waits'].append(s))
    return state


@pytest.mark.parametrize('code', [413, 429, 500, 503])
def test_transient_codes_are_retried_with_the_same_request(transport, code):
    transport['script'] = [http_error(code), http_error(code), 'ok']
    text, meta = semantic.call_responses(OPTIONS, [{'role': 'user', 'content': []}])
    assert text.startswith('view=front') and meta['attempts'] == 3 and transport['waits'] == [5, 15]


@pytest.mark.parametrize('code', [400, 401, 403, 404])
def test_other_client_errors_fail_at_once(transport, code):
    transport['script'] = [http_error(code), 'ok']
    with pytest.raises(RuntimeError, match='HTTP %d' % code):
        semantic.call_responses(OPTIONS, [])
    assert transport['calls'] == 1 and transport['waits'] == []


def test_413_that_never_clears_still_ends_in_an_error_after_all_attempts(transport):
    transport['script'] = [http_error(413)] * 4
    with pytest.raises(RuntimeError, match='HTTP 413.*after 4 attempts'):
        semantic.call_responses(OPTIONS, [])
    assert transport['calls'] == 4 and transport['waits'] == [5, 15, 45]


def test_run_accepts_no_images_and_defaults_to_off():
    from occlusion import cli
    args = cli.parser().parse_args(['run', '--model', 'omninav', '--batch', 'b'])
    assert args.no_images is False
    args = cli.parser().parse_args(['run', '--model', 'omninav', '--batch', 'b', '--no-images'])
    assert args.no_images is True
