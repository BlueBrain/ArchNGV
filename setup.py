#!/usr/bin/env python

import imp
from setuptools import setup
from setuptools import find_packages

VERSION = imp.load_source("archngv.version", "archngv/version.py").VERSION


EXTRA_CORE = [
    'tess>=0.2.2',
    'tmd>=2.0.6',
    'MorphIO>=2.3.4',
    'numpy-stl>=2.7',
    'trimesh>=2.21.15',
    'pandas>=0.16.2',
    'spatial-index==0.0.1',
    'archngv-building>=0.1.3',
]

EXTRA_ALL = [
    'bluepy[sonata]>=0.13.5',
    'Click>=7.0',
    'openmesh==1.1.2',
    'pyyaml>=3.0',
    # TODO 'tns>=??',
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
        'libsonata>=0.1.1',
        'h5py>=2.3.1',
        'cached-property>=1.5',
        'voxcell[sonata]>=2.6.2',
        'scipy>=1.0.0'
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
