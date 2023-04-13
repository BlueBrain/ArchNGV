#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR=build

rm -rf $BUILD_DIR && mkdir $BUILD_DIR

pushd $BUILD_DIR

unset $(env | grep SLURM | cut -d= -f1 | xargs)

ngv run \
    --bioname '../bioname' \
    --snakefile '../../../archngv/app/snakemake/Snakefile' \
    --cluster-config '../bioname/cluster.yaml'

popd

pytest test_specifications.py
pytest test_build.py
pytest test_circuit_integrity.py
