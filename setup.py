"""Install in each model environment without moving the source directory."""
from setuptools import find_packages, setup

setup(
    name='vln-intervene',
    version='0.1.0',
    description='Deviation-point occlusion interventions for VLN models',
    python_requires='>=3.6',
    packages=['intervene'] + ['intervene.' + name for name in find_packages(exclude=('tests', 'tests.*', 'intervene', 'intervene.*'))],
    package_dir={'intervene': 'intervene', 'intervene.occlusion': 'occlusion'},
    package_data={'intervene.occlusion': ['configs/*.template.json', 'configs/*.fragment.json']},
    install_requires=['numpy>=1.19'],
    extras_require={'images': ['Pillow'], 'test': ['pytest']},
    entry_points={'console_scripts': ['intervene-occlusion=intervene.occlusion.cli:main']},
)
