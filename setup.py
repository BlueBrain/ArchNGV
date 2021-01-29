#!/usr/bin/env python

import importlib.util
from setuptools import setup
from setuptools import find_packages


spec = importlib.util.spec_from_file_location("archngv.version", "archngv/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


BUILDING = [
    'archngv-building>=0.1.3',
    'bluepy-configfile>=0.1.11',
    'Click>=7.0',
    'diameter-synthesis>=0.1.7',
    'MorphIO>=2.3.4',
    'morph-tool>=0.2.10',
    'numpy-stl>=2.7',
    'openmesh>=1.1.2',
    'pyyaml>=3.0',
    'pandas<1.1.0',  # py-touchreader spack module overwrites numpy version to 1.15.2, making pandas throw
    'spatial-index==0.0.2',
    'tess>=0.2.2',
    'tmd>=2.0.6',
    'tns>=2.2.1',
    'trimesh>=2.21.15'
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
        'numpy>=1.15.4',
    ],
    install_requires=[
        'numpy>=1.15.4',
        'six>=1.15.0',
        'h5py>=3.1.0',
        'scipy>=1.0.0',
        'libsonata>=0.1.1',
        'bluepysnap>=0.9.0',
        'cached-property>=1.5',
        'voxcell>=3.0.0',
        'vasculatureapi>=0.0.6',
    ],
    extras_require={
        'all': BUILDING,
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
