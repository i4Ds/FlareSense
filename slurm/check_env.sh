#!/bin/sh
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --partition=top6
#SBATCH --out=slurm/log_out.txt
#SBATCH --err=slurm/log_err.txt
#SBATCH --job-name="Check Environment"
python3 -m pip install -r requirements.txt
echo

python3 -c "import torch; print(f'GPUs: {[torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]}')"
python3 -c "import os; print(f'Logical cores (threads): {os.cpu_count()}')"
echo

nvidia-smi
echo

python3 -m pip freeze