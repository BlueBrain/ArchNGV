#!/bin/bash

rm -f out*
rm -f err*

SCRIPT="$PWD/../archngv_workflow/workflow.py"
CONFIG="$PWD/../test_data/test_config.json"
LAUNCH="$PWD/launch.sbatch"
ARGUMENTS="${@:1}"

echo $ARGUMENTS

#sbatch $LAUNCH $SCRIPT $PWD/test_config.json --seed=0 --repeat=0 --run_all
python $SCRIPT $CONFIG --seed=0 "${ARGUMENTS[@]}"
