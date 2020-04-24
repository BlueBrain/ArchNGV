#!/bin/bash

# SYNOPSIS
#    ./create_exemplars.sh dir [exemplars]
#
# DESCRIPTION
#    Create circuit exemplars in the specifed directory
#    based on a template located in ./template directory.
#
#    dir        root output directory.
#
#    exemplars  number of exemplars to create, default is 7.

ARCHNGV_PATH=$(dirname $(dirname $(readlink -fm "$0")))
ARCHNGV_EXEMPLAR="$ARCHNGV_PATH/exemplar/template"

TARGET_PARENT_DIR=$1
NUM_EXEMPLARS="${2:-7}"

for INDEX in `seq 0 $(($NUM_EXEMPLARS - 1))`
do

    TARGET_DIR="$TARGET_PARENT_DIR/exemplar_$INDEX"

    mkdir -p $TARGET_DIR

    ARCHNGV_SNAKEFILE="$ARCHNGV_PATH/snakemake/Snakefile"
    TARGET_SNAKEFILE="$TARGET_DIR/Snakefile"

    ARCHNGV_BIONAME="$ARCHNGV_EXEMPLAR/bioname"
    TARGET_BIONAME="$TARGET_DIR/bioname"

    ln -s $ARCHNGV_BIONAME $TARGET_BIONAME
    ln -s $ARCHNGV_SNAKEFILE $TARGET_SNAKEFILE
    ln -s "$ARCHNGV_EXEMPLAR/run.sh" "$TARGET_DIR/run.sh" && chmod +x "$TARGET_DIR/run.sh"

done
