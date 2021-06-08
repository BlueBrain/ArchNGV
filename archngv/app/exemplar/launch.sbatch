#!/bin/bash
#
#
#SBATCH --partition prod
#SBATCH --account proj62
#SBATCH --ntasks=36
#SBATCH --nodes 1
#SBATCH --time 1-00:00:00
#SBATCH --job-name sNGV
#SBATCH --output out-%J.log
#SBATCH --error err-%J.log
##SBATCH --mem 300000
##SBATCH --exclusive

##set -e
##set -x

PROJECT_NAME=$1
PROJECT_PATH="$PWD/$PROJECT_NAME"
VENV_PATH=$2

echo $(hostname)

module purge
module load archive/2021-05 python/3.8.3 cmake

source VENV_PATH

pushd $(mktemp -d)

sh ${PROJECT_PATH}/run.sh

popd
