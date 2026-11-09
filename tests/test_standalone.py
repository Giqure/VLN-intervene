"""Check isolated input paths and a simulator-free end-to-end report."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from occlusion.core.intervention import Intervention
from occlusion.core.rig import Frame, Pose, Rig, View


REPO = Path(__file__).resolve().parents[1]


def isolated_python(script, directory, overrides=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('INTERVENE_')}
    env['PYTHONPATH'] = str(REPO)
    env.update(overrides or {})
    return json.loads(subprocess.check_output([sys.executable, '-c', script], cwd=str(directory),
                                             env=env, universal_newlines=True))


def test_external_paths_and_model_imports_are_isolated(tmp_path):
    names = ('WORKSPACE_ROOT', 'MODELS_ROOT', 'DATASET_ROOT', 'EXPORT_ROOT',
             'ARCHIVE_ROOT', 'RESULTS_ROOT', 'ENV_FILE')
    overrides = {'INTERVENE_' + name: str(tmp_path / name.lower()) for name in names}
    output = isolated_python(
        "import json, sys; from occlusion import paths, runner; "
        "from occlusion.core import store, semantic; from occlusion.select import common; "
        "from occlusion.viz import report; "
        "print(json.dumps(dict(workspace=str(paths.WORKSPACE), models=str(common.OPEN), "
        "dataset=str(common.DATASET), export=str(common.EXPORT), archive=str(store.ARCHIVE), "
        "results=str(runner.RESULTS), report_results=str(report.RESULTS), "
        "env_file=str(semantic.ENV_FILE), "
        "loaded=[n for n in ('torch', 'habitat', 'habitat_sim', 'vlnce_baselines') if n in sys.modules])))",
        tmp_path, overrides)
    for field, name in [('workspace', 'WORKSPACE_ROOT'), ('models', 'MODELS_ROOT'),
                        ('dataset', 'DATASET_ROOT'), ('export', 'EXPORT_ROOT'),
                        ('archive', 'ARCHIVE_ROOT'), ('results', 'RESULTS_ROOT'),
                        ('report_results', 'RESULTS_ROOT'), ('env_file', 'ENV_FILE')]:
        assert output[field] == overrides['INTERVENE_' + name]
    assert output['loaded'] == []


def test_legacy_layout_and_repo_local_reports(tmp_path):
    output = isolated_python(
        "import json; from occlusion import paths; "
        "print(json.dumps(dict(repo=str(paths.REPO_ROOT), workspace=str(paths.WORKSPACE), "
        "models=str(paths.OPEN), results=str(paths.RESULTS))))", tmp_path)
    assert output['repo'] == str(REPO)
    assert output['workspace'] == str(REPO.parent)
    assert output['models'] == str(REPO.parent.parent / 'opensource')
    assert output['results'] == str(REPO / 'results')


def test_reports_use_configured_roots_without_models(tmp_path):
    output = isolated_python(
        """
import json
from pathlib import Path
from occlusion import runner
from occlusion.core import store
from occlusion.viz.report import build
p = json.loads((Path(%r) / 'occlusion/configs/omninav-rgb.template.json').read_text())
p.update(depth_bias=0.01, depth_max=10.0)
p['adapter']['history'] = 'clean'
batch = store.batch_name(p)
store.ARCHIVE.mkdir()
root = runner.batch_dir('omninav', batch)
root.mkdir(parents=True)
(root / 'params.json').write_text(json.dumps(p))
(root / 'candidates.jsonl').write_text('')
summary = runner.analyze('omninav', batch, n_boot=10)
report = build('omninav', batch)
print(json.dumps(dict(report=report['report'], complete=summary['checks']['complete'],
                      summary=str(runner.RESULTS / 'omninav' / batch / 'summary.json'))))
""" % str(REPO), tmp_path,
        {'INTERVENE_ARCHIVE_ROOT': str(tmp_path / 'archive'),
         'INTERVENE_RESULTS_ROOT': str(tmp_path / 'reports')})
    assert output['complete'] == 0
    assert Path(output['report']).is_file()
    assert Path(output['summary']).is_file()
    assert str(tmp_path / 'reports') in output['report']


def test_blackout_preserves_clean_observations_and_follow_limit():
    rig = Rig('synthetic', [View('front', 0.0, 90.0, (5, 7), (5, 7))],
              1.0, 10.0, 1.5, 'wxyz', 0.01)
    tracker = dict(kind='heading', occlusion=False, rows='axis', incremental=False, unseen_step=None)
    hook = Intervention(dict(step=2, center=0.0), dict(kind='heading', half=15.0),
                        tracker, follow=0, channels=['rgb'], rig=rig)
    rgb = np.full((5, 7, 3), 255, dtype=np.uint8)
    depth = np.full((5, 7), 5.0)
    pose = Pose(np.zeros(3), 0.0)
    before = Frame(1, pose, {'front': rgb}, {'front': depth})
    assert hook.on_obs(before) is before
    changed = hook.on_obs(Frame(2, pose, {'front': rgb}, {'front': depth}))
    assert np.all(changed.rgb['front'][:, 3] == 0)
    assert np.all(changed.rgb['front'][:, 0] == 255)
    assert np.all(rgb == 255)
    assert np.all(depth == 5.0)
    after = Frame(3, pose, {'front': rgb}, {'front': depth})
    assert hook.on_obs(after) is after
    assert [row['step'] for row in hook.log()] == [2]
