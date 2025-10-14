#!/bin/bash

#SBATCH -N 2
#SBATCH -n 2
#SBATCH --job-name SVM&KNN

srun -n1 --exclusive python3 cluster_ml2csv_all1.py &
srun -n1 --exclusive python3 cluster_ml2csv_all2.py &
wait