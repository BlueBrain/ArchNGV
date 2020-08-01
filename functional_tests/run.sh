#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR=build

rm -rf $BUILD_DIR && mkdir $BUILD_DIR

pushd $BUILD_DIR

unset $(env | grep SLURM | cut -d= -f1 | xargs)

snakemake \
    --snakefile '../../snakemake/Snakefile' --cluster-config '../bioname/cluster.yaml'\
    --config bioname='../bioname' \
    --cores 5 \

popd

./compare.sh expected $BUILD_DIR
