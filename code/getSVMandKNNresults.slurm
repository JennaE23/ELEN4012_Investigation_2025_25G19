#!/bin/bash

#SBATCH -N 4
#SBATCH -n 8
#SBATCH --job-name SVM&KNN

echo "Running jobs"
srun -n1 --exclusive python3 code/cluster_ml2csv_all1.py &
srun -n1 --exclusive python3 code/cluster_ml2csv_all2.py &
srun -n1 --exclusive python3 code/cluster_ml2csv_all3.py &
srun -n1 --exclusive python3 code/cluster_ml2csv_all4.py &
srun -n1 --exclusive python3 code/cluster_ml2csv_all5.py &
srun -n1 --exclusive python3 code/cluster_ml2csv_all6.py &
srun -n1 --exclusive python3 code/cluster_ml2csv_all7.py &
srun -n1 --exclusive python3 code/cluster_ml2csv_all8.py &
wait
