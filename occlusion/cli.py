"""Run with python -m intervene.occlusion in the selected model's environment."""
import argparse
import json
import sys
from pathlib import Path

from . import runner
from .core import store


def parser():
    p = argparse.ArgumentParser(description='C6002 deviation-point occlusion experiments')
    sub = p.add_subparsers(dest='command')
    plan = sub.add_parser('plan', help='read original outputs and annotations, never forward a model')
    plan.add_argument('--config', required=True, type=Path, help='JSON batch parameters (all geometry explicit)')
    plan.add_argument('--none-batch', help='fully reproduced ETPNav none batch for ghost provenance')
    for name in ('run', 'analyze', 'viz', 'offline'):
        command = sub.add_parser(name)
        command.add_argument('--model', choices=sorted(runner.ADAPTERS), required=True)
        command.add_argument('--batch', required=True)
        if name in ('run', 'offline'):
            command.add_argument('--episode', action='append', help='repeat to restrict episodes')
            command.add_argument('--limit', type=int)
        if name == 'run':
            command.add_argument('--fail-fast', action='store_true')
            command.add_argument('--retry-errors', action='store_true',
                                 help='repeat candidates whose earlier attempt failed (only needed for SimpleVLN; '
                                      'OmniNav and ETPNav always rerun them)')
        if name == 'analyze':
            command.add_argument('--n-boot', type=int, default=1000)
            command.add_argument('--level', type=float, default=.95)
            command.add_argument('--seed', type=int, default=0)
        if name == 'offline':
            command.add_argument('--check-rgb', action='store_true', help='compare baseline/augmented sensors pixel-for-pixel')
    meta = sub.add_parser('commit-meta', help='single-process archive-root metadata maintenance')
    meta.add_argument('--model', choices=sorted(runner.ADAPTERS), action='append', required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.command:
        p.print_help()
        return 2
    if getattr(args, 'limit', None) is not None and args.limit < 1:
        p.error('--limit must be positive')
    try:
        if args.command == 'plan':
            params = json.loads(args.config.resolve().read_text(encoding='utf-8'))
            result = runner.plan(params, args.none_batch)
        elif args.command == 'run':
            result = runner.run(args.model, args.batch, args.episode, args.limit, args.fail_fast, args.retry_errors)
        elif args.command == 'analyze':
            if not 0 < args.level < 1 or args.n_boot < 1:
                p.error('require 0 < level < 1 and n-boot > 0')
            result = runner.analyze(args.model, args.batch, args.n_boot, args.level, args.seed)
        elif args.command == 'viz':
            from .viz.report import build
            result = build(args.model, args.batch)
        elif args.command == 'offline':
            from .offline import render
            result = render(args.model, args.batch, args.episode, args.limit, args.check_rgb)
        else:
            with runner.exclusive(store._archive_available() / '.c6002-meta.lock'):
                store.commit_meta(args.model)
            result = {'metadata': 'committed', 'models': args.model}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get('errors') else 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 1
