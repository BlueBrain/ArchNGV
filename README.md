# Usage on BB5

## Installation

```shell
# Clone this repository
$ git clone ssh://bbpcode.epfl.ch/molecularsystems/ArchNGV /path/to/repo

# Load most recent stable modules
$ module load `date +%Y-%m`

# Load modules required to build ArchNGV
$ module load cmake python snakemake

# Create a Python virtualenv in repository source directory
$ python -m venv /path/to/repo/.venv

# Bring the virtualenv in this shell environment
$ . /path/to/repo/.venv/bin/activate

# Install ArchNGV
$ cd /path/to/repo
$ make install
```

## Create circuit exemplars

```shell
# Create a directory for your circuit
$ circuit_dir=/gpfs/bbp.cscs.ch/project/projXX/$USER/ArchNGVCircuits
$ mkdir -p $circuit_dir

# Create an exemplar
$ python ./exemplar/create_exemplar.py $circuit_dir
```

## Execute cell placement

To proceed to the cell placement in one of the created exemplar:

```
# Change directory to one of the created exemplar
$ cd $circuit_dir/exemplar_ID

# Execute the "cell-placement" snakemake target
$ ./run.sh cell-placement
# -> creates file build/cell_data.h5
```

Use the `cell_data_sonata` SnakeMake task to perform output conversion to Sonata format
after the cell placement:

```shell
$ ./run.sh cell_data_sonata

# sonata file glia.h5.somata is created in the sonata.tmp directory
$ find build/sonata.tmp
build/sonata.tmp
build/sonata.tmp/nodes
build/sonata.tmp/nodes/glia.h5.somata
```
