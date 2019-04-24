#!/usr/bin/env bash

#SBATCH --job-name=preprocessing_patrec
#SBATCH --time=168:00:00
#SBATCH --qos=1week
#SBATCH --mem=60G
#SBATCH --mem-per-cpu=60G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-user="t.sutter@unibas.ch"

DIRPROJECTBASE=$HOME/projects/PATREC
DIRDATA=$DIRPROJECTBASE/data
DIRCODE=$DIRPROJECTBASE/PATREC

source $DIRPROJECTBASE/setup.sh

python mainPreprocessing.py