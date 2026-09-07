#!/bin/bash

#SBATCH --job-name=Logistic Regression
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=1000Gb
#SBATCH --time=366:59:59
#SBATCH --mail-type=end
#SBATCH --mail-user=email@txstate.edu

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."

# Move to the correct directory

cd /home/user/Network_Paper
echo "Current working directory is `pwd`"

# Train & Test the model
python 1_Logistic_Regression.py

# end of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
