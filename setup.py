#!/usr/bin/env python

import importlib.util
from setuptools import setup
from setuptools import find_packages


spec = importlib.util.spec_from_file_location("archngv.version", "archngv/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


BUILDING = [
    # py{38,39,310} wheels for >=0.1.6
    # pinned to 0.1.6 because the next version will be implemented in c++
    'ngv-ctools==0.1.6',
    'spatial-index>=0.4.2',
    'bluepy-configfile>=0.1.11',
    'Click>=7.0',
    'numpy-stl>=2.10',
    'openmesh>=1.1.2',
    'pyyaml>=5.0',
    'pandas>=1.1.0',
    'tess>=0.3.1',
    # archive/2021-05 touchdetector crashes for >3.3.0 due to hdf5 file version incompatibility
    'MorphIO>=3.0.0,<=3.3.0',
    'morph-tool>=2.4.0',
    'pytouchreader>=1.4.7',
    'tmd>=2.0.11',
    'tns==2.4.3',
    'diameter-synthesis==0.2.4',
    'trimesh>=3.9.9',
    # constrain dask dependencies <=2.21 according to their BB5 deployed versions
    'dask[distributed,bag]>=2.0,<=2.21',
    'distributed>=2.0,<=2.21',
    'dask_mpi>=2.0,<=2.21',
]


setup(
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
    name='ArchNGV',
    version=VERSION,
    description='NGV Architecture Modules',
    author='Eleftherios Zisis',
    author_email='eleftherios.zisis@epfl.ch',
    setup_requires=['cython'],  # needed to correctly compile tess by generating new c files
    url="https://bbpteam.epfl.ch/documentation/projects/ArchNGV",
    project_urls={
        "Tracker": "",
        "Source": "https://bbpgitlab.epfl.ch/molsys/ArchNGV.git",
    },
    license="BBP-internal-confidential",
    install_requires=[
        'numpy>=1.19.5,<1.22.0',  # functional tests break for >=1.22.0
        'six>=1.15.0',
        'h5py>=3.1.0',
        'scipy>=1.5.0',
        'libsonata>=0.1.8',
        "bluepysnap>=0.13,<1.0",
        'cached-property>=1.5',
        'voxcell>=3.0.0',
        'vasculatureapi>=0.3.0',
    ],
    extras_require={
        'all': BUILDING,
        'docs': ['sphinx', 'sphinx-bluebrain-theme']
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
