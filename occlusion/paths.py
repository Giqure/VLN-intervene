"""External input paths; overrides are read once when the process starts.

Defaults preserve the original VLN layout. Standalone clones should configure
their external inputs explicitly; none of these paths creates or copies data.
"""
import os
from pathlib import Path


def _path(name, default):
    value = os.environ.get(name)
    if value is not None and not value.strip():
        raise ValueError('%s must not be empty' % name)
    return Path(os.path.expandvars(value if value is not None else str(default))).expanduser().resolve()


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = _path('INTERVENE_WORKSPACE_ROOT', REPO_ROOT.parent)
OPEN = _path('INTERVENE_MODELS_ROOT', WORKSPACE.parent / 'opensource')
DATASET = _path('INTERVENE_DATASET_ROOT', WORKSPACE.parent / 'dataset/R2R_VLNCE_v1-3_preprocessed')
EXPORT = _path('INTERVENE_EXPORT_ROOT', WORKSPACE / 'results/export')
ARCHIVE = _path('INTERVENE_ARCHIVE_ROOT', '/media/jwm/大数据/result')
RESULTS = _path('INTERVENE_RESULTS_ROOT', REPO_ROOT / 'results')
ENV_FILE = _path('INTERVENE_ENV_FILE', REPO_ROOT / '.env' if (REPO_ROOT / '.env').is_file()
                 else WORKSPACE / '.env')
