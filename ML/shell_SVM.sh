#!/bin/bash
#SBATCH --account=def-maxwl 
#SBATCH --mem-per-cpu=70000M
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=3
 

python3 SVM.py| tee log.txt
