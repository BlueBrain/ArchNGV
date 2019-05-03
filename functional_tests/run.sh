#!/usr/bin/env bash

set -euo pipefail

BUILD_DIR=build
NGV_CONFIG=$BUILD_DIR/ngv_config.json

rm -rf $BUILD_DIR

mkdir $BUILD_DIR
mkdir $BUILD_DIR/intermediate
mkdir $BUILD_DIR/morphology

sed 's;$BASE_DIR;'$(pwd)';g' ngv_config.json.template > $NGV_CONFIG

python ../archngv/workflow/apps/ngv_input_generation.py $NGV_CONFIG \
    --neuron_synapse_connectivity

python ../archngv/workflow/apps/ngv_main_workflow.py $NGV_CONFIG \
    --cell_placement \
    --microdomains \
    --gliovascular_connectivity \
    --neuroglial_connectivity \
    --synthesis

./compare.sh expected $BUILD_DIR
