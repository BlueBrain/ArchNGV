#!/bin/bash
#
#
#SBATCH --partition prod
#SBATCH --account proj62
#SBATCH --nodes 1
#SBATCH --time 1-00:00:00
#SBATCH --job-name sNGV
#SBATCH --output out-%J.log
#SBATCH --error err-%J.log
#SBATCH --mem 300000
#SBATCH -Cnvme
##SBATCH --exclusive
##SBATCH --nice=500


##set -e
##set -x

PYTHONSCRIPT=$1
SCRIPTARGUMENTS="${@:2}"

echo $(hostname)

source /gpfs/bbp.cscs.ch/project/proj62/NGV/workflow_modules.sh
source /gpfs/bbp.cscs.ch/project/proj62/NGV/workflow_venv/bin/activate

pushd $(mktemp -d)

echo $PWD

python $PYTHONSCRIPT $SCRIPTARGUMENTS

popd
