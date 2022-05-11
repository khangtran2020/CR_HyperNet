#!/bin/bash -l
# The above line must always be first, and must have "-l"
#SBATCH -J hpn_udp
#SBATCH -p datasci
#SBATCH --output= results/clean_bt_80_1000_step.out
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
module load python
conda activate hypernet
python userdp_hypernet_cifar10.py --train_mode clean --grad_clip 1.0 --delta 0.00001 --noise_scale 1.5 --num_comp_cli 1 --bt 80 --num_steps 1000

