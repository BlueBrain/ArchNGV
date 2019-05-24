#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR=build

rm -rf $BUILD_DIR && mkdir $BUILD_DIR

pushd $BUILD_DIR

snakemake \
	--snakefile '../../snakemake/Snakefile' \
	--config bioname='../bioname' \
	-- default sonata

popd

./compare.sh expected $BUILD_DIR
