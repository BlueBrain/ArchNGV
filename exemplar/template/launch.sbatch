#!/bin/bash
#
#
#SBATCH --partition prod
#SBATCH --account proj62
#SBATCH --ntasks=72
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

echo $(hostname)

module purge
module load archive/2019-12 python/3.6.5 cmake/3.13.0
source /gpfs/bbp.cscs.ch/project/proj62/NGV/ARCHNGV/PROJECTS/endfeet_paper/venv/bin/activate

pushd $(mktemp -d)

snakemake --snakefile $PROJECT_PATH/Snakefile \
          --config bioname=$PROJECT_PATH/bioname \
          --directory $PROJECT_PATH/build \
          --cluster-config $PROJECT_PATH/bioname/cluster.yaml \
          -p 72 \
          -F all

popd
