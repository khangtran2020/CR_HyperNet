#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J hpn_clean
#SBATCH -p datasci
#SBATCH --output=results/hpn_clean_bt_80_step_1000.out
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
module load python
conda activate hypernet
python main.py --train_mode clean --grad_clip 1.0 --delta 0.00001 --noise_scale 1.5 --num_comp_cli 1 --bt 80 --num_steps 1000

