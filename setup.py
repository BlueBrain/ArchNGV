#!/usr/bin/env python

import os
from setuptools import setup
from setuptools import Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as build_ext


NAME='archngv'

APPS_FILENAMES = ['ngv_initialize_directories.py',
                  'ngv_input_generation.py',
                  'ngv_main_workflow.py',
                  'ngv_preprocessing.py',
                  'ngv_postprocessing.py']

APPS = ["{}/workflow/apps/{}".format(NAME, p) for p in APPS_FILENAMES]


pwd = os.path.dirname(__file__)


SETUP_REQUIREMENTS = \
[
    'numpy>=1.15',
    'cython>=0.25.2'
]


REQUIREMENTS = \
[
    'cached-property>=1.3.1',
    'enum34>=1.0.4',
    'h5py>=2.3.1',
    'jenkspy>=0.1.4',
    'lxml>=4.1.1',
    'openmesh>=1.1.2',
    'pandas>=0.16.2',
    'numpy-stl>=2.7',
    'Rtree>=0.8.3',
    'scipy>=1.1.0',
    'tess>=0.2.2',
    'trimesh>=2.21.15'
] + [
    'morphmath>=0.0',
    'morphspatial>=0.0',
    'spatial_index>=0.0',
    'voxcell>=2.5.2'
]


DEPENDENCIES = \
[
    'git+ssh://git@github.com/eleftherioszisis/MorphMath.git#egg=morphmath-0.0',
    'git+ssh://git@github.com/eleftherioszisis/MorphSpatial.git#egg=morphspatial-0.0',
    'git+ssh://git@github.com/eleftherioszisis/SpatialIndex.git#egg=spatial_index-0.0',
    'https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/voxcell#egg=voxcell-2.5.2'
]


def scandir(directory, files=[]):

    for f in os.listdir(directory):

        f_path = os.path.join(directory, f)

        if os.path.isfile(f_path) and f_path.endswith(".pyx"):
            files.append(f_path)
        elif os.path.isdir(f_path):
            scandir(f_path, files)

    return files

def create_extensions(directory):

    extensions = []

    for fpath in scandir(directory):

        dpath = fpath.replace(os.path.sep, '.')[:-4]
        
        extension = Extension(
             dpath,
             sources=[fpath],
             #include_dirs=[numpy.get_include()],
             language='c++',
             extra_compile_args=["-O2"]
        )
        extensions.append(extension)

    return extensions

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


setup(
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

      name=NAME,

      version='0.0',

      description = 'NGV Architecture Modules',

      author='Eleftherios Zisis',

      author_email = 'eleftherios.zisis@epfl.ch',

      setup_requires = SETUP_REQUIREMENTS,

      install_requires = REQUIREMENTS,

      dependency_links = DEPENDENCIES,

      packages = find_packages(),

      scripts = APPS,

      cmdclass = {'build_ext': CustomBuildExtCommand},

      ext_modules = create_extensions(NAME),

      include_package_data = True
     )

