#!/usr/bin/env bash

JOBNAME="training_neuralnetwork_PATREC"

#SBATCH --job-name=$JOBNAME
#SBATCH --qos=6hours
#SBATCH --time=02:00:00
#SBATCH --mem=5G
#SBATCH --tmp=10G            # the compute node should have at least 10G of free space in local scratch folder ($TMPDIR)
#SBATCH --mail-user="t.sutter@unibas.ch"

DIRPROJECT=$HOME/projects/PATREC
DIRDATA=$DIRPROJECT/data
DIRCODE=$DIRPROJECT/PATREC

module purge
echo "install python..."
module load  Python/3.5.2-goolf-1.7.20
echo "DONE!"
#echo "install tensorflow..."
#module load Tensorflow/1.4.1-goolf-1.7.20-Python-3.5.2
#echo "DONE"

echo "source virtualenv..."
cp -r $DIRPROJECT/venv-test $TMPDIR
source $TMPDIR/venv-test/bin/activate
echo "DONE!"

#cp -r $DIRDATA/* $TMPDIR
cp $DIRDATA/data_nz_20122016_standard_embedding_grouping.csv $TMPDIR/

python mainNeuralNetworkTraining.py $TMPDIR

cp -r $TMPDIR/* $HOME/output-data/
rm $TMPDIR/data_20122015_embedding_ready_*