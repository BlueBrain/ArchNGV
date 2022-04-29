#!/usr/bin/env python

import importlib.util

from setuptools import find_packages, setup

spec = importlib.util.spec_from_file_location("archngv.version", "archngv/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


BUILDING = [
    "ngv-ctools>=1.0.0",
    "spatial-index>=0.4.2",
    "bluepy-configfile>=0.1.11",
    "Click>=7.0",
    "numpy-stl>=2.10,<2.16.0",  # More recent versions require >1.22.0 which is not in spack yet
    "openmesh>=1.1.2",
    "pyyaml>=5.0",
    "pandas>=1.1.0",
    # tess has been mirrored in bbpgitlab: https://bbpgitlab.epfl.ch/nse/mirrors/tess
    # and wheels have been released in devpi for >=0.3.2
    "tess==0.3.2",
    "MorphIO>=3.2.0",
    "morph-tool>=2.4.0",
    "pytouchreader>=1.4.7",
    "snakemake>=5.0.0",
    "tmd>=2.0.11",
    # latest version available in spack before OSS to NeuroTS
    # we cannot yet switch to neurots because there are major breaking changes
    # that will require rebuilding the distribution inputs to circuit building
    "tns==2.5.0",
    # available version in spack, compatible with tns==2.5.0
    "diameter-synthesis==0.2.5",
    "dask[distributed,bag]>=2.0",
    # 2022.4.1 crashes dask parallel
    "distributed!=2022.4.1",
    "dask_mpi>=2.0",
]


setup(
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    name="ArchNGV",
    version=VERSION,
    description="NGV Architecture Modules",
    author="Eleftherios Zisis",
    author_email="eleftherios.zisis@epfl.ch",
    url="https://bbpteam.epfl.ch/documentation/projects/ArchNGV",
    project_urls={
        "Tracker": "",
        "Source": "https://bbpgitlab.epfl.ch/molsys/ArchNGV.git",
    },
    license="BBP-internal-confidential",
    install_requires=[
        "numpy>=1.19.5",
        "h5py>=3.1.0",
        "scipy>=1.5.0",
        "libsonata>=0.1.8",
        "bluepysnap>=0.13,<1.0",
        "cached-property>=1.5",
        "voxcell>=3.0.0",
        "vascpy>=0.1.0",
        # 2.38.10 is available in spack.
        # Update to a higher version in spack after checking that py-atlas-building-tools works
        "trimesh>=2.38.10",
    ],
    extras_require={"all": BUILDING, "docs": ["sphinx", "sphinx-bluebrain-theme"]},
    packages=find_packages(),
    scripts=[],
    entry_points={"console_scripts": ["ngv=archngv.app.__main__:app"]},
    include_package_data=True,
)
