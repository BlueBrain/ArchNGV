#!/usr/bin/env python

import os
import imp
from setuptools import setup
from setuptools import Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as build_ext


VERSION = imp.load_source("archngv.version", "archngv/version.py").VERSION

def create_extensions(directory):

    def scandir(directory, files=[]):

        for f in os.listdir(directory):

            f_path = os.path.join(directory, f)

            if os.path.isfile(f_path) and f_path.endswith(".pyx"):
                files.append(f_path)
            elif os.path.isdir(f_path):
                scandir(f_path, files)

        return files

    extensions = []

    for filepath in scandir(directory):

        dotted_path = filepath.replace(os.path.sep, '.')[:-4]
        extension = Extension(dotted_Path, sources=[filepath], language='c++', extra_compile_args=["-O2"])
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

      name = 'archngv',

      version = VERSION,

      description = 'NGV Architecture Modules',

      author ='Eleftherios Zisis',

      author_email = 'eleftherios.zisis@epfl.ch',

      setup_requires = [
                            'numpy>=1.13',
                            'cython>=0.25.2'
      ],

      install_requires = [
                            'morphmath',
                            'morphspatial',
                            'spatial_index',
                            'cached-property>=1.3.1',
                            'enum34>=1.0.4',
                            'h5py>=2.3.1',
                            'jenkspy>=0.1.4',
                            'openmesh>=1.1.2',
                            'pandas>=0.16.2',
                            'numpy-stl>=2.7',
                            'scipy>=1.0.0',
                            'tess>=0.2.2',
                            'trimesh>=2.21.15'
      ],

      packages = find_packages(),

      scripts = [
                  'archngv/workflow/apps/ngv_initialize_directories.py',
                  'archngv/workflow/apps/ngv_input_generation.py',
                  'archngv/workflow/apps/ngv_main_workflow.py',
                  'archngv/worfklow/apps/ngv_preprocessing.py',
                  'archngv/workflow/apps/ngv_postprocessing.py'
      ],

      cmdclass = {'build_ext': CustomBuildExtCommand},

      ext_modules = create_extensions('archngv'),

      include_package_data = True
)
