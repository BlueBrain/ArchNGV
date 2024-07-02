
.. image: https://raw.githubusercontent.com/BlueBrain/ArchNGV/main/doc/source/logo/logo.jpg

# ArchNGV
Modules for in-silco building the Neuronal-Glial-Vascular structural architecture.

## Documentation:

* [latest](https://archngv.readthedocs.io/en/latest)

* [File Format Specification](https://sonata-extension.readthedocs.io/en/latest/index.html)

## Installation

### From PyPI

```shell
pip install archngv[all]
```

### From source
```shell
# Clone this repository
git clone https://github.com/BlueBrain/ArchNGV.git

# Create a Python virtualenv in repository source directory
python -m venv /path/to/repo/.venv

# Bring the virtualenv in this shell environment
. /path/to/repo/.venv/bin/activate

# Install ArchNGV
cd ArchNGV/
pip install .[all]
```
# Examples
## Create circuit exemplars

```shell
# Create a directory for your circuit
circuit_dir=./circuits
mkdir -p $circuit_dir

# Create an exemplar
python ./exemplar/create_exemplar.py $circuit_dir
```

## Execute cell placement

To proceed to the cell placement in one of the created exemplar:

```
# Change directory to one of the created exemplar
cd $circuit_dir/exemplar_ID

# Execute the "cell-placement" snakemake target
./run.sh cell-placement
# -> creates file build/cell_data.h5
```

Use the `cell_data_sonata` SnakeMake task to perform output conversion to Sonata format
after the cell placement:

```shell
./run.sh cell_data_sonata

# sonata file glia.h5.somata is created in the sonata.tmp directory
find build/sonata.tmp
# build/sonata.tmp
# build/sonata.tmp/nodes
# build/sonata.tmp/nodes/glia.h5.somata
```

## Astrocyte Synthesis

It uses Dask for parallel calculations.

An example for a local machine usage. Pay attention that `--parallel` option is not used:
```shell
ngv synthesis \
    --config /path/to/synthesis_config.yaml' \
    --tns-distributions /path/to/tns_distributions.json \
    --tns-parameters /path/to/tns_parameters.json \
    --tns-context /path/to/tns_context.json \
    --astrocytes /path/to/glia.h5 \
    --microdomains /path/to/microdomains.h5 \
    --gliovascular-connectivity /path/to/gliovascular.h5 \
    --neuroglial-connectivity /path/to/neuroglial.h5 \
    --endfeet-areas /path/to/endfeet_areas.h5 \
    --neuronal-connectivity /path/to/edges.h5 \
    --out-morph-dir /path/to/out_morphologies
```
An example for a BB5 usage. `--parallel` option is used in conjunction with multiple exclusive
nodes, otherwise no benefits from `--parallel`:
```shell
srun -Aproj<your_project> -N2 -t=24:00:00 --exclusive \
ngv synthesis \
    --config ... \
    // all above options as before
    --parallel
```
An example of Snakemake usage. 
```shell
snakemake \
    --snakefile snakemake/Snakefile \
    --cluster-config path/to/cluster.yaml \  # see an example below
    --config bioname path/to/bioname \
    synthesis
```

Don't forget to have an entry in your `cluster.yaml` for `synthesis`
```yaml
synthesis:
  jobname: ngv_synthesis
  account: '<your project>'
  nodes: <better have at least 2, 4 is recommended>
  partition: 'prod'
  constraint: 'cpu'
  time: '04:00:00' # feel free to increase time 
  cpus-per-task: 2
  exclusive: true
  mem: 0
```

Also it is highly recommended to export in advance next Dask variables for better performance:
```shell
export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80  # pause execution at 80% memory use
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # restart the worker at 95% use
# Reduce dask profile memory usage/leak (see https://github.com/dask/distributed/issues/4091)
export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=10000ms  # Time between statistical profiling queries
export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=1000000ms  # Time between starting new profile
```

A final sbatch script example
```shell
#!/bin/bash
# below SBATCH options are not related to options from cluster.yaml 
#SBATCH --partition prod
#SBATCH --account proj62
#SBATCH --nodes 1  # It is not recommended to set higher because synthesis task will be launched with SBATCH options from its entry in `cluster.yaml`  
#SBATCH --time 08:00:00
#SBATCH --job-name sNGV
#SBATCH --output out-%J.log
#SBATCH --error err-%J.log
#SBATCH --mem 200000
#SBATCH --exclusive
export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80  # pause execution at 80% memory use
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # restart the worker at 95% use
# Reduce dask profile memory usage/leak (see https://github.com/dask/distributed/issues/4091)
export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=10000ms  # Time between statistical profiling queries
export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=1000000ms  # Time between starting new profile

source <venv with ArchNGV installed>/bin/activate

snakemake --snakefile <path to Snakefile of this project> \
          --config bioname=<path to bioname> \
          --directory <path to save results> \
          --cluster-config <path to your cluster.yaml> \
          -f synthesis
```
## Citation
When you use ArchNGV software or methods in your research, we ask you to cite the following publication (this includes poster presentations):

[Zisis E, Keller D, Kanari L, Arnaudon A, Gevaert M, Delemontex T, Coste B, Foni A, Abdellah M, Calì C, Hess K, Magistretti PJ, Schürmann F, Markram H. 2021. Digital Reconstruction of the Neuro-Glia-Vascular Architecture. Cerebral Cortex. 31:5686–5703.
](https://doi.org/10.1093/cercor/bhab254)

```
@article{10.1093/cercor/bhab254,
    author = {Zisis, Eleftherios and Keller, Daniel and Kanari, Lida and Arnaudon, Alexis and Gevaert, Michael and Delemontex, Thomas and Coste, Benoît and Foni, Alessandro and Abdellah, Marwan and Calì, Corrado and Hess, Kathryn and Magistretti, Pierre Julius and Schürmann, Felix and Markram, Henry},
    title = "{Digital Reconstruction of the Neuro-Glia-Vascular Architecture}",
    journal = {Cerebral Cortex},
    volume = {31},
    number = {12},
    pages = {5686-5703},
    year = {2021},
    month = {08},
    issn = {1047-3211},
    doi = {10.1093/cercor/bhab254},
    url = {https://doi.org/10.1093/cercor/bhab254},
    eprint = {https://academic.oup.com/cercor/article-pdf/31/12/5686/40814577/bhab254.pdf},
}
```

## Acknowledgements

This publication is based upon work supported by the King Abdullah University of Science and Technology (KAUST) Office of Sponsored Research (OSR) under Award No. OSR-2017-CRG6-3438

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2019-2024 Blue Brain Project/EPFL
