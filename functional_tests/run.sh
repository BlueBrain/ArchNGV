#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR=build

module load nix/py36/snakemake

rm -rf $BUILD_DIR && mkdir $BUILD_DIR

pushd $BUILD_DIR

snakemake --snakefile '../Snakefile' --config bioname='../bioname'

popd

./compare.sh expected $BUILD_DIR
