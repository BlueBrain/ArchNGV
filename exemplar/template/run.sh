#!/bin/bash

# SYNOPSIS
#    ./run.sh [task]
#
# DESCRIPTION
#     Execute the SnakeMake tasks described in this directory.
#
#     task  SnakeMake task name, default is "all".

snakemake --snakefile ./Snakefile \
          --config bioname=../bioname \
          --directory ./build \
          --cluster-config ./bioname/cluster.yaml \
          -F "${1:-all}"
