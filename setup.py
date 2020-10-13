#!/usr/bin/env python

import importlib.util
from setuptools import setup
from setuptools import find_packages


spec = importlib.util.spec_from_file_location("archngv.version", "archngv/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


EXTRA_CORE = [
    'tess>=0.2.2',
    'tmd>=2.0.6',
    'MorphIO>=2.3.4',
    'numpy-stl>=2.7',
    'trimesh>=2.21.15',
    'spatial-index==0.0.1',
    'archngv-building>=0.1.3',
]

EXTRA_ALL = [
    'Click>=7.0',
    'openmesh>=1.1.2',
    'pyyaml>=3.0',
    'tns>=2.2.1',
    'diameter-synthesis>=0.1.6',
    'morph-tool>=0.2.10'
]

setup(
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    name='archngv',
    version=VERSION,
    description='NGV Architecture Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    setup_requires=[
        'numpy>=1.13',
    ],
    install_requires=[
        'pandas==1.0.5',
        'bluepy-configfile>=0.1.11',
        'six>=1.15.0',
        'h5py>=2.3.1',
        'scipy>=1.0.0',
        'libsonata>=0.1.1',
        'bluepysnap>=0.6.1',
        'cached-property>=1.5',
        'voxcell[sonata]>=2.6.2',
        'vasculatureapi>=0.0.6',
    ],
    extras_require={
        'all': EXTRA_CORE + EXTRA_ALL,
        'core': EXTRA_CORE,
    },
    packages=find_packages(),
    scripts=[
    ],
    entry_points={
        'console_scripts': [
            'ngv=archngv.app.__main__:app'
        ]
    },
    include_package_data=True
)
