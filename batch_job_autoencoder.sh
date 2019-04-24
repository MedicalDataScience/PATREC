#!/usr/bin/env bash

#SBATCH --job-name=autoencoder_nz
#SBATCH --time=168:00:00
#SBATCH --qos=1week
#SBATCH --mem=60G
#SBATCH --mem-per-cpu=60G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-user="t.sutter@unibas.ch"

DIRPROJECTBASE=$HOME/projects/PATREC
DIRDATA=$DIRPROJECTBASE/data
DIRCODE=$DIRPROJECTBASE/PATREC
DIROUTPUT=$HOME/nz_autoencoder_cbow

DIRPRETRAIN=$DIROUTPUT/autoencoder_nz_20122016_reduction_FUSION_embedding_verylightgrouping_16_dropout_None_learningrate_0.01_batchnorm_True_batchsize_160

source $DIRPROJECTBASE/setup.sh

#echo "start copying stuff (data/scripts)..."
#cp -r $DIRDATA/* $TMPDIR
#cp -r $DIRDATA/"data_nz_20122016_reduction_FUSION_embedding_verylightgrouping.csv" $TMPDIR/
#cp -r $DIRCODE/ $TMPDIR/
mkdir $DIROUTPUT
pwd

echo "start training..."
python mainAutoEncoderTraining.py --hidden_units=16 --learningrate=0.01 --batch_size=160 --train_epochs=250 --model_dir=$DIROUTPUT/ --continue_training=True --pretrained_model_dir=$DIRPRETRAIN/