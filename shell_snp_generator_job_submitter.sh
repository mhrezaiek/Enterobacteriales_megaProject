#!/bin/bash
#SBATCH --account=def-maxwl 
#SBATCH --mem-per-cpu=700M
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=8
 

python3 Dataset_downloader.py| tee log.txt
