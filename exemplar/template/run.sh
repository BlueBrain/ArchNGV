#!/bin/bash

# SYNOPSIS
#    ./run.sh [task]
#
# DESCRIPTION
#     Execute the SnakeMake tasks described in this directory.
#
#     task  SnakeMake task name, default is "all".

export DASK_DISTRIBUTED__WORKER__USE_FILE_LOCKING=False
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=False  # don't spill to disk
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.80  # pause execution at 80% memory use
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.95  # restart the worker at 95% use
# Reduce dask profile memory usage/leak (see https://github.com/dask/distributed/issues/4091)
export DASK_DISTRIBUTED__WORKER__PROFILE__INTERVAL=10000ms  # Time between statistical profiling queries
export DASK_DISTRIBUTED__WORKER__PROFILE__CYCLE=1000000ms  # Time between starting new profile

snakemake --snakefile ./Snakefile \
          --config bioname=../bioname \
          --directory ./build \
          --cluster-config ./bioname/cluster.yaml \
          --cores 5 \
          -f "${1:-all}"
