#!/usr/bin/env python

import importlib.util

from setuptools import find_packages, setup

spec = importlib.util.spec_from_file_location("archngv.version", "archngv/version.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION


BUILDING = [
    "ngv-ctools>=1.0.0",
    "Click>=7.0",
    "numpy-stl>=2.10",
    "openmesh>=1.1.2",
    "pyyaml>=5.0",
    "pandas>=1.1.0",
    "multivoro",
    "MorphIO>=3.2.0",
    "morph-tool>=2.4.0",
    "snakemake>=5.0.0",
    "tmd>=2.0.11",
    "NeuroTS>=3.4.0",
    "diameter-synthesis>=0.5.4",
    "dask[distributed,bag]>=2022.04.1",
    "dask_mpi>=2.0",
    "spatial-index>=2.0.0",
    "atlas-commons>=0.1.4",
    "meshio>=5.3.4",
]


setup(
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
        "numpy>=1.22",
        "h5py>=3.1.0",
        "scipy>=1.5.0",
        # 0.1.21 supports resolution of ngv populations extra files
        "libsonata>=0.1.21",
        "bluepysnap>=1.0",
        "cached-property>=1.5",
        "voxcell>=3.0.0",
        "vascpy>=0.1.0",
        # 3.17.1 is available in spack.
        # Update to a higher version in spack after checking that py-atlas-building-tools works
        "trimesh>=3.17.1",  # Method Trimesh:subdivide_loop is needed
        # trimesh soft dependency for using marching cubes
        "scikit-image>=0.18",
        # requests module pin urrlib3 in recent versions
        # see https://github.com/psf/requests/issues/6432
        "urllib3>=1.21.1,<1.27; python_version <= '3.8'",
    ],
    extras_require={"all": BUILDING, "docs": ["sphinx", "sphinx-bluebrain-theme"]},
    packages=find_packages(),
    scripts=[],
    entry_points={"console_scripts": ["ngv=archngv.app.__main__:app"]},
    include_package_data=True,
)
