#!/bin/sh
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=slurm/log_out.txt
#SBATCH --err=slurm/log_err.txt
#SBATCH --job-name="FlareSense"
python3 -m pip install -r requirements.txt
echo

dvc pull
cd notebooks
python3 -m papermill 00-Unzip.ipynb 00-Unzip.ipynb -k 'python3'
echo

NB_PATH="06-ResNet18.ipynb"
python3 -m papermill $NB_PATH $NB_PATH -k 'python3'

echo "Done!"