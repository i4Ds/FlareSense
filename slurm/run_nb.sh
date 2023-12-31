#!/bin/sh
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=performance
#SBATCH --out=slurm/log_out.txt
#SBATCH --err=slurm/log_err.txt
#SBATCH --job-name="FlareSense"
#SBATCH --exclude=gpu22a,gpu22b,node15,sdas2
python3 -m pip install -r requirements.txt
echo

#dvc pull
cd notebooks
#python3 -m papermill 00-Unzip.ipynb 00-Unzip.ipynb -k 'python3'
#echo

NB_PATH="10-ResNet50.ipynb"
python3 -m papermill $NB_PATH $NB_PATH -k 'python3'

echo "Done!"