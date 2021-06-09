How to create a new ngv circuit build
=====================================

Preparation
===========


1. Load the necessary modules

.. code-block:: bash

    module purge
    module load archive/2021-05 python/3.8.3 cmake

2. Create and source a virtual environment

.. code-block:: bash

    python3.8 -m venv venv3.8
    source venv3.8

3. Install the ngv building pipeline

.. code-block:: bash

    git clone https://bbpgitlab.epfl.ch/molsys/ArchNGV.git
    cd ArchNGV/
    make install

4. Create an ngv circuit exemplar

.. code-block:: bash

    ngv create-exemplar project_dir

Which will create the ``project_dir`` directory with a ``bioname`` folder, which includes default recipes for building an ngv circuit.


Running
=======

The ``ngv`` command will appear to the path following the package installation. The ``ngv run`` command is a wrapper around snakemake with default preset options and arguments. It uses the same options and arguments as snakemake. For example:

.. code-block:: bash

    ngv run --bioname /path/to/bioname --cluster-config /path/to/cluster.yaml

where the ``cluster.yaml`` is located inside the ``bioname`` folder for the exemplar above.

`archngv/snakemake/Snakefile` contains many ready to use recipes that allow you to generate any part of an ngv circuit. Such recipes are called phases in snakemake. Usually it is a name of a circuit file you want to generate. To choose a phase, type its name at the end of circuit-build run call. For example:

.. code-block:: bash

    ngv run --bioname /path/to/bioname --cluster-config /path/to/cluster.yaml phase-name

