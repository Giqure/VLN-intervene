"""Package entry for editable installs with the existing occlusion/ layout."""
from pathlib import Path

_source = Path(__file__).resolve().parent.parent
if (_source / 'setup.py').is_file() and (_source / 'occlusion/__init__.py').is_file():
    __path__.append(str(_source))
