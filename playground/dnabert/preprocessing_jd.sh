#!/bin/bash
#$ -cwd
#$ -o joblog.$JOB_ID
#$ -j y

#$ -l h_rt=24:00:00,h_data=32G
#$ -pe shared 2

#$ -M $USER@mail
#$ -m bea

echo "Job $JOB_ID started on: " `hostname -s`
echo "Start time: " `date`
echo " "

# Load module system
. /u/local/Modules/default/init/modules.sh

# Load required modules
module load python/3.9.6
module load anaconda3

# Activate your conda env
source ~/.bashrc
conda activate ernst_env

echo "Python location:"
which python
python --version

# Go to your folder
cd ~/dnabert

echo "Running script"
# Using python3 to ensure f-strings and underscores work
/usr/bin/time -v python preprocessing_dnabert.py

echo " "
echo "Job $JOB_ID ended on: " `hostname -s`
echo "End time: " `date`
echo " "
