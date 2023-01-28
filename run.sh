#!/bin/bash
#SBATCH --job-name=FA_clipping
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=8gb
#SBATCH --cpus-per-task=4

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=outfiles/%j.err
#SBATCH --output=outfiles/%j.out
##SBATCH --partition=AMD6276

#SBATCH --array=1-5
echo "using scavenger"

# Prepare conda:
eval "$(conda shell.bash hook)"
conda activate /home/jacob.adamczyk001/miniconda3/envs/oblenv
export CPATH=$CPATH:$CONDA_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

echo "Start Run"
echo `date`
# python BoundingValue/experiment.py --algo 'bound'
python uchi_agent_NN.py

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"